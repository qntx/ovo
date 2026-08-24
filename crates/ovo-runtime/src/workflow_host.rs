//! Adapter: `ovo-workflow` host channel → [`SessionHost`] nested runs.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ovo_obs::{NoopMetrics, SharedMetrics, record_workflow_agents, record_workflow_run};
use ovo_tools::EventBus;
use ovo_tools::registry::CapabilityMode;
use ovo_workflow::{
    AgentOpts, AgentResult, BudgetState, DEFAULT_AGENT_BUDGET, HostError, WorkflowHostRequest,
    WorkflowOutcome, WorkflowRunParams, run_workflow,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{Instrument, info_span};

use crate::host::{SessionHost, SpawnOpts};
use crate::side_effects::WorkflowSideEffects;

/// Run a workflow script whose `agent` / `parallel` calls resolve through `host`.
///
/// Blocks the calling async task on a worker thread for the Rhai engine while
/// servicing host requests on the current runtime.
///
/// `agent_budget` of `None` applies [`DEFAULT_AGENT_BUDGET`] (128), not
/// unlimited. This adapter gate is a second counter, independent of the host
/// spawn budget.
///
/// # Errors
///
/// Propagates channel / join failures as [`HostError::Failed`]. Workflow
/// terminal outcomes are returned as [`WorkflowOutcome`] (including
/// `Failed` / `BudgetExceeded` variants) rather than `Err`.
pub async fn run_workflow_on_host(
    host: Arc<dyn SessionHost>,
    params: WorkflowRunParams,
    agent_budget: Option<u64>,
) -> Result<WorkflowOutcome, HostError> {
    run_workflow_on_host_with_metrics(host, params, agent_budget, Arc::new(NoopMetrics)).await
}

/// Like [`run_workflow_on_host`] with an explicit metrics sink.
///
/// # Errors
///
/// Same as [`run_workflow_on_host`].
pub async fn run_workflow_on_host_with_metrics(
    host: Arc<dyn SessionHost>,
    params: WorkflowRunParams,
    agent_budget: Option<u64>,
    metrics: SharedMetrics,
) -> Result<WorkflowOutcome, HostError> {
    run_workflow_configured_with_events(
        host,
        params,
        agent_budget,
        metrics,
        WorkflowSideEffects::shared(),
        None,
    )
    .await
}

/// Full configuration: metrics + side-effect store (scratch / templates).
///
/// # Errors
///
/// Same as [`run_workflow_on_host`].
pub async fn run_workflow_configured(
    host: Arc<dyn SessionHost>,
    params: WorkflowRunParams,
    agent_budget: Option<u64>,
    metrics: SharedMetrics,
    effects: Arc<WorkflowSideEffects>,
) -> Result<WorkflowOutcome, HostError> {
    run_workflow_configured_with_events(host, params, agent_budget, metrics, effects, None).await
}

/// Like [`run_workflow_configured`] with an optional live [`EventBus`] for nested spawns.
///
/// Mode B `agent()` / `parallel()` emit the same `SpawnStarted` / `SpawnFinished`
/// shapes as Mode A when `events` is set.
///
/// `agent_budget` of `None` applies [`DEFAULT_AGENT_BUDGET`] (128), not unlimited.
///
/// # Errors
///
/// Same as [`run_workflow_on_host`].
pub async fn run_workflow_configured_with_events(
    host: Arc<dyn SessionHost>,
    mut params: WorkflowRunParams,
    agent_budget: Option<u64>,
    metrics: SharedMetrics,
    effects: Arc<WorkflowSideEffects>,
    events: Option<EventBus>,
) -> Result<WorkflowOutcome, HostError> {
    let (tx, mut rx) = mpsc::unbounded_channel::<WorkflowHostRequest>();
    let spent = Arc::new(AtomicU64::new(0));
    let reserved = Arc::new(AtomicU64::new(0));
    let cancel = params.cancel.clone();

    let budget = Some(agent_budget.unwrap_or(DEFAULT_AGENT_BUDGET));
    let spent_h = Arc::clone(&spent);
    let reserved_h = Arc::clone(&reserved);
    let cancel_h = cancel.clone();
    let host_svc = Arc::clone(&host);
    let effects_svc = Arc::clone(&effects);

    let service = tokio::spawn(async move {
        service_loop(
            &mut rx,
            host_svc,
            budget,
            spent_h,
            reserved_h,
            cancel_h,
            effects_svc,
            events,
        )
        .await;
    });

    params.host_tx = tx;
    let outcome = tokio::task::spawn_blocking(move || run_workflow(params))
        .await
        .map_err(|e| HostError::Failed(format!("workflow join: {e}")))?;

    // Dropping the sender (inside run_workflow when it finishes) ends the service loop.
    let _ = service.await;

    let spent_n = spent.load(Ordering::Relaxed);
    record_workflow_agents(metrics.as_ref(), spent_n);
    record_workflow_run(metrics.as_ref(), outcome_label(&outcome));
    Ok(outcome)
}

#[allow(
    clippy::too_many_arguments,
    reason = "service loop owns host channel state"
)]
async fn service_loop(
    rx: &mut mpsc::UnboundedReceiver<WorkflowHostRequest>,
    host_svc: Arc<dyn SessionHost>,
    budget: Option<u64>,
    spent_h: Arc<AtomicU64>,
    reserved_h: Arc<AtomicU64>,
    cancel_h: CancellationToken,
    effects_svc: Arc<WorkflowSideEffects>,
    events: Option<EventBus>,
) {
    let mut inflight = Vec::new();
    while let Some(req) = rx.recv().await {
        if cancel_h.is_cancelled() {
            reply_cancelled(req);
            continue;
        }
        match req {
            WorkflowHostRequest::SpawnAgent { opts, reply } => {
                let host = Arc::clone(&host_svc);
                let spent = Arc::clone(&spent_h);
                let reserved = Arc::clone(&reserved_h);
                let cancel = cancel_h.clone();
                let events = events.clone();
                inflight.push(tokio::spawn(async move {
                    handle_spawn(
                        host.as_ref(),
                        opts,
                        reply,
                        &spent,
                        &reserved,
                        &cancel,
                        events.as_ref(),
                    )
                    .await;
                }));
            }
            other => {
                dispatch_inline(other, budget, &spent_h, &reserved_h, effects_svc.as_ref());
            }
        }
    }
    for t in inflight {
        let _ = t.await;
    }
}

fn outcome_label(outcome: &WorkflowOutcome) -> &'static str {
    match outcome {
        WorkflowOutcome::Completed { .. } => "completed",
        WorkflowOutcome::Paused { .. } => "paused",
        WorkflowOutcome::BudgetExceeded { .. } => "budget_exceeded",
        WorkflowOutcome::Cancelled => "cancelled",
        WorkflowOutcome::Failed { .. } => "failed",
        _ => "other",
    }
}

async fn handle_spawn(
    host: &dyn SessionHost,
    opts: AgentOpts,
    reply: tokio::sync::oneshot::Sender<Result<AgentResult, HostError>>,
    spent: &AtomicU64,
    reserved: &AtomicU64,
    cancel: &CancellationToken,
    events: Option<&EventBus>,
) {
    let span = info_span!(
        "ovo.workflow.host",
        ovo.workflow.kind = "spawn_agent",
        ovo.agent_label = opts.label.as_deref().unwrap_or(""),
    );
    let result = async {
        if cancel.is_cancelled() {
            return Err(HostError::Cancelled);
        }
        let spawn = to_spawn_opts(opts, cancel.child_token(), events)?;
        match host.spawn_agent(spawn).await {
            Ok(run) => {
                spent.fetch_add(1, Ordering::Relaxed);
                let r = reserved.load(Ordering::Relaxed);
                reserved.fetch_sub(r.min(1), Ordering::Relaxed);
                let tokens = u64::from(run.usage.total_tokens);
                Ok(AgentResult {
                    agent_id: run.agent_id.to_string(),
                    success: run.success && !run.cancelled,
                    output: run.output,
                    cancelled: run.cancelled,
                    tokens_used: tokens,
                    duration_ms: run.duration_ms,
                })
            }
            Err(e) => {
                let r = reserved.load(Ordering::Relaxed);
                reserved.fetch_sub(r.min(1), Ordering::Relaxed);
                Err(map_host_spawn_error(e))
            }
        }
    }
    .instrument(span)
    .await;
    let _ = reply.send(result);
}

fn dispatch_inline(
    req: WorkflowHostRequest,
    budget: Option<u64>,
    spent: &AtomicU64,
    reserved: &AtomicU64,
    effects: &WorkflowSideEffects,
) {
    match req {
        WorkflowHostRequest::ReserveAgentCalls { count, reply } => {
            let result = reserve(budget, spent, reserved, count);
            let _ = reply.send(result);
        }
        WorkflowHostRequest::ReleaseAgentCalls { count, reply } => {
            let r = reserved.load(Ordering::Relaxed);
            reserved.fetch_sub(count.min(r), Ordering::Relaxed);
            let _ = reply.send(Ok(()));
        }
        WorkflowHostRequest::SpawnAgent { reply, .. } => {
            // Concurrent path is handled in the service loop.
            let _ = reply.send(Err(HostError::Failed(
                "internal: SpawnAgent must be handled concurrently".into(),
            )));
        }
        WorkflowHostRequest::BudgetQuery { reply } => {
            let s = spent.load(Ordering::Relaxed);
            let r = reserved.load(Ordering::Relaxed);
            let state = BudgetState {
                total: budget,
                spent: s,
                reserved: r,
                remaining: budget.map(|b| b.saturating_sub(s.saturating_add(r))),
            };
            let _ = reply.send(Ok(state));
        }
        WorkflowHostRequest::Phase { title, replayed } => {
            tracing::info!(target: "ovo.workflow", %title, replayed, "phase");
        }
        WorkflowHostRequest::Log { message, replayed } => {
            tracing::info!(target: "ovo.workflow", %message, replayed, "log");
        }
        WorkflowHostRequest::Telemetry {
            name,
            fields,
            replayed,
        } => {
            tracing::info!(target: "ovo.workflow", %name, %fields, replayed, "telemetry");
        }
        WorkflowHostRequest::RenderTemplate { reply, name, vars } => {
            let _ = reply.send(effects.render_template(&name, &vars));
        }
        WorkflowHostRequest::WriteScratchFile {
            reply,
            name,
            content,
        } => {
            let _ = reply.send(effects.write_scratch(&name, content));
        }
        WorkflowHostRequest::ReadScratchFile { reply, name } => {
            let _ = reply.send(effects.read_scratch(&name));
        }
        WorkflowHostRequest::GitDiffSince { reply, commit } => {
            let _ = reply.send(effects.git_diff_since(&commit));
        }
    }
}

fn reserve(
    budget: Option<u64>,
    spent: &AtomicU64,
    reserved: &AtomicU64,
    count: u64,
) -> Result<(), HostError> {
    if let Some(max) = budget {
        loop {
            let s = spent.load(Ordering::Acquire);
            let r = reserved.load(Ordering::Acquire);
            if s.saturating_add(r).saturating_add(count) > max {
                return Err(HostError::AgentCallQuotaExceeded {
                    requested: s.saturating_add(r).saturating_add(count),
                    maximum: max,
                });
            }
            if reserved
                .compare_exchange(
                    r,
                    r.saturating_add(count),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return Ok(());
            }
        }
    }
    reserved.fetch_add(count, Ordering::Relaxed);
    Ok(())
}

/// Map workflow [`AgentOpts`] → host [`SpawnOpts`] without silent field drops.
///
/// `fork_context` requires host `parent_handle` or explicit `fork_messages`.
/// `resume_from` requires host `run_store` and a completed [`WorkflowRunStore`] row.
fn map_host_spawn_error(e: ovo_types::OvoError) -> HostError {
    use ovo_types::ErrorCode;
    match e.code() {
        ErrorCode::HostBudget => HostError::BudgetExceeded,
        ErrorCode::HostCancelled => HostError::Cancelled,
        ErrorCode::HostUnsupported
        | ErrorCode::HostDepth
        | ErrorCode::HostConcurrency
        | ErrorCode::AgentNotFound => HostError::Unsupported(e.message().to_owned()),
        _ => HostError::Failed(e.to_string()),
    }
}

fn to_spawn_opts(
    opts: AgentOpts,
    cancel: CancellationToken,
    events: Option<&EventBus>,
) -> Result<SpawnOpts, HostError> {
    let mut spawn = SpawnOpts::new(opts.prompt).with_cancel(cancel);
    if let Some(label) = opts.label {
        spawn = spawn.with_label(label);
    }
    if let Some(model) = opts.model {
        spawn.model = Some(model);
    }
    if let Some(mode) = opts.capability_mode.as_deref() {
        spawn.capability_mode = parse_capability(mode)?;
    }
    if let Some(agent_type) = opts.agent_type {
        spawn = spawn.with_agent_type(agent_type);
    }
    if let Some(schema) = opts.output_schema {
        spawn = spawn.with_output_schema(schema);
    }
    if let Some(n) = opts.max_output_tokens {
        spawn = spawn.with_max_output_tokens(n);
    }
    if opts.fork_context {
        spawn = spawn.with_fork_context(true);
    }
    if let Some(id) = opts.resume_from {
        spawn = spawn.with_resume_from(id);
    }
    if let Some(bus) = events {
        spawn = spawn.with_events(bus.clone());
    }
    Ok(spawn)
}

fn parse_capability(mode: &str) -> Result<CapabilityMode, HostError> {
    CapabilityMode::parse(mode).ok_or_else(|| {
        HostError::Unsupported(format!(
            "unknown capability_mode '{mode}' (expected full|read_only|plan)"
        ))
    })
}

fn reply_cancelled(req: WorkflowHostRequest) {
    match req {
        WorkflowHostRequest::ReserveAgentCalls { reply, .. }
        | WorkflowHostRequest::ReleaseAgentCalls { reply, .. } => {
            let _ = reply.send(Err(HostError::Cancelled));
        }
        WorkflowHostRequest::SpawnAgent { reply, .. } => {
            let _ = reply.send(Err(HostError::Cancelled));
        }
        WorkflowHostRequest::BudgetQuery { reply } => {
            let _ = reply.send(Err(HostError::Cancelled));
        }
        WorkflowHostRequest::RenderTemplate { reply, .. }
        | WorkflowHostRequest::WriteScratchFile { reply, .. }
        | WorkflowHostRequest::ReadScratchFile { reply, .. }
        | WorkflowHostRequest::GitDiffSince { reply, .. } => {
            let _ = reply.send(Err(HostError::Cancelled));
        }
        WorkflowHostRequest::Phase { .. }
        | WorkflowHostRequest::Log { .. }
        | WorkflowHostRequest::Telemetry { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ovo_llm::MockSampler;
    use ovo_workflow::{Journal, WorkflowOutcome, WorkflowRunParams};
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::host::InProcessHost;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn workflow_spawn_emits_events_on_bus() {
        use ovo_protocol::TurnEventKind;
        use ovo_tools::EventBus;
        use ovo_types::RunId;

        let sampler = Arc::new(MockSampler::new());
        sampler.push_text("ok");
        let host: Arc<dyn SessionHost> = Arc::new(InProcessHost::new(sampler, vec![]));
        let script = r#"
            let meta = #{ name: "ev", description: "test" };
            let r = agent("task", #{ label: "child" });
            complete(#{ out: r });
        "#;
        let (tx, mut rx) = mpsc::unbounded_channel();
        let bus = EventBus::new(tx, RunId::generate());
        let (host_tx, _host_rx) = mpsc::unbounded_channel();
        let outcome = run_workflow_configured_with_events(
            host,
            WorkflowRunParams {
                script: script.into(),
                args: serde_json::json!({}),
                journal: Journal::new(None),
                host_tx,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            },
            Some(4),
            Arc::new(NoopMetrics),
            WorkflowSideEffects::shared(),
            Some(bus),
        )
        .await
        .expect("workflow");
        assert!(
            matches!(outcome, WorkflowOutcome::Completed { .. }),
            "expected completed, got {outcome:?}"
        );

        let mut saw_start = false;
        let mut saw_finish = false;
        let mut finish_ok = true;
        while let Ok(ev) = rx.try_recv() {
            saw_start |= matches!(ev.kind, TurnEventKind::SpawnStarted { .. });
            if let TurnEventKind::SpawnFinished { success, .. } = ev.kind {
                saw_finish = true;
                finish_ok &= success;
            }
        }
        assert!(
            saw_start && saw_finish && finish_ok,
            "mode B must emit spawn lifecycle"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn workflow_parallel_on_session_host() {
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("a", "from-a");
        sampler.map_user_text("b", "from-b");
        let host: Arc<dyn SessionHost> = Arc::new(InProcessHost::new(sampler, vec![]));
        let script = r#"
            let meta = #{ name: "fanout", description: "test" };
            phase("work");
            let rs = parallel([
                #{ prompt: "a", label: "wa" },
                #{ prompt: "b", label: "wb" },
            ]);
            complete(#{ results: rs });
        "#;
        let (tx, _rx) = mpsc::unbounded_channel();
        let outcome = run_workflow_on_host(
            host,
            WorkflowRunParams {
                script: script.into(),
                args: serde_json::json!({}),
                journal: Journal::new(None),
                host_tx: tx,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            },
            Some(16),
        )
        .await
        .expect("run");
        let WorkflowOutcome::Completed { result } = outcome else {
            unreachable!("expected completed outcome");
        };
        let arr = result
            .get("results")
            .and_then(|v| v.as_array())
            .expect("results array");
        assert_eq!(arr.len(), 2);
        assert_eq!(
            arr.first().and_then(|v| v.get("output")),
            Some(&serde_json::json!("from-a"))
        );
        assert_eq!(
            arr.get(1).and_then(|v| v.get("output")),
            Some(&serde_json::json!("from-b"))
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn workflow_budget_on_host() {
        let sampler = Arc::new(MockSampler::new());
        sampler.push_text("x");
        let host: Arc<dyn SessionHost> =
            Arc::new(InProcessHost::new(sampler, vec![]).with_agent_budget(0));
        // Budget 0 at adapter reserve layer
        let script = r#"
            let meta = #{ name: "b", description: "b" };
            agent("x");
            complete(1);
        "#;
        let (tx, _rx) = mpsc::unbounded_channel();
        let outcome = run_workflow_on_host(
            host,
            WorkflowRunParams {
                script: script.into(),
                args: serde_json::json!({}),
                journal: Journal::new(None),
                host_tx: tx,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            },
            Some(0),
        )
        .await
        .expect("run");
        assert!(
            matches!(outcome, WorkflowOutcome::BudgetExceeded { .. }),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn workflow_none_budget_is_default_128() {
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("task", "ok");
        let host: Arc<dyn SessionHost> = Arc::new(InProcessHost::new(sampler, vec![]));
        let script = r#"
            let meta = #{ name: "n", description: "none budget" };
            let r = agent("task");
            complete(#{ out: r });
        "#;
        let (tx, _rx) = mpsc::unbounded_channel();
        let outcome = run_workflow_on_host(
            host,
            WorkflowRunParams {
                script: script.into(),
                args: serde_json::json!({}),
                journal: Journal::new(None),
                host_tx: tx,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            },
            None,
        )
        .await
        .expect("run");
        assert!(
            matches!(outcome, WorkflowOutcome::Completed { .. }),
            "{outcome:?}"
        );
    }
}
