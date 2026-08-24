//! Session host for nested agent runs (dynamic multi-agent delegation).
//!
//! # Limits (fail-closed)
//!
//! - **`agent_budget`** — admitted spawns for this host. [`InProcessHost::new`]
//!   defaults to [`ovo_workflow::DEFAULT_AGENT_BUDGET`] (128);
//!   [`InProcessHost::with_agent_budget`] caps at
//!   [`ovo_workflow::MAX_AGENT_BUDGET`] (1024). Unlimited only via
//!   [`InProcessHost::with_unlimited_agent_budget`].
//! - **`max_spawn_depth`** — max nesting index (`depth` on [`SpawnOpts`]); `depth >= max` rejects
//! - **`max_concurrent_children`** — simultaneous in-flight spawns (try-acquire; no queue)

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use async_trait::async_trait;
use futures::future::try_join_all;
use ovo_agent::{
    Agent, AgentBuilder, AgentDefinition, AgentRegistry, IdentityAssembler, PromptAssembler,
};
use ovo_llm::LlmSampler;
use ovo_obs::{NoopMetrics, SharedMetrics, record_spawn};
use ovo_protocol::TurnEventKind;
use ovo_state::ChatStateHandle;
use ovo_tools::registry::CapabilityMode;
use ovo_tools::{ApprovalGate, ApprovalPolicy, EventBus, SharedTool};
use ovo_types::{AgentId, ErrorCode, Message, OvoError, Usage};
use ovo_workflow::{WorkflowRunStatus, WorkflowRunStore};
use serde_json::Value;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;
use tracing::{Instrument, Span, info_span};

use crate::isolation::{InProcessIsolation, IsolationBackend, IsolationEnv};
use crate::state::VecConversationState;
use crate::turn::{TurnInput, TurnOptions, TurnRuntime};

/// Default max nesting depth for nested agents (`0..DEFAULT_MAX_SPAWN_DEPTH`).
pub const DEFAULT_MAX_SPAWN_DEPTH: u32 = 16;
/// Default max concurrent in-flight nested agents.
pub const DEFAULT_MAX_CONCURRENT_CHILDREN: usize = 64;

/// Options for spawning a nested agent.
///
/// Field parity with workflow [`ovo_workflow::AgentOpts`] for Mode A/B isomorphism.
#[derive(Debug, Clone)]
pub struct SpawnOpts {
    /// User prompt for the child.
    pub prompt: String,
    /// Optional label for logs / result correlation.
    pub label: Option<String>,
    /// Override model.
    pub model: Option<String>,
    /// Capability mode for child tools.
    pub capability_mode: CapabilityMode,
    /// Max steps for child turn.
    pub max_steps: Option<usize>,
    /// Cancel token for this child.
    pub cancel: CancellationToken,
    /// Definition name resolved via host agent catalogue.
    pub agent_type: Option<String>,
    /// Optional JSON schema for structured child output.
    pub output_schema: Option<Value>,
    /// When true, seed the child conversation from [`Self::fork_messages`].
    pub fork_context: bool,
    /// Parent messages injected when [`Self::fork_context`] is true.
    pub fork_messages: Option<Vec<Message>>,
    /// Replay a completed workflow run id via host [`WorkflowRunStore`] (charges budget).
    pub resume_from: Option<String>,
    /// Max output tokens hint for the child sample.
    pub max_output_tokens: Option<u64>,
    /// Nesting depth of this spawn (`0` = first level under the host).
    pub depth: u32,
    /// Parent turn event bus (spawn lifecycle events use this stream).
    pub events: Option<EventBus>,
}

impl SpawnOpts {
    /// Prompt-only spawn with a fresh cancel token at depth 0.
    #[must_use]
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            label: None,
            model: None,
            capability_mode: CapabilityMode::Full,
            max_steps: None,
            cancel: CancellationToken::new(),
            agent_type: None,
            output_schema: None,
            fork_context: false,
            fork_messages: None,
            resume_from: None,
            max_output_tokens: None,
            depth: 0,
            events: None,
        }
    }

    /// Set label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set capability mode.
    #[must_use]
    pub const fn with_capability(mut self, mode: CapabilityMode) -> Self {
        self.capability_mode = mode;
        self
    }

    /// Set cancel token (often a child of a parent token).
    #[must_use]
    pub fn with_cancel(mut self, cancel: CancellationToken) -> Self {
        self.cancel = cancel;
        self
    }

    /// Set max steps.
    #[must_use]
    pub const fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    /// Set nesting depth.
    #[must_use]
    pub const fn with_depth(mut self, depth: u32) -> Self {
        self.depth = depth;
        self
    }

    /// Set agent type / definition name.
    #[must_use]
    pub fn with_agent_type(mut self, agent_type: impl Into<String>) -> Self {
        self.agent_type = Some(agent_type.into());
        self
    }

    /// Set structured output schema for the child.
    #[must_use]
    pub fn with_output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Set max output tokens hint.
    #[must_use]
    pub const fn with_max_output_tokens(mut self, n: u64) -> Self {
        self.max_output_tokens = Some(n);
        self
    }

    /// Request parent conversation fork (requires [`Self::with_fork_messages`]).
    #[must_use]
    pub const fn with_fork_context(mut self, fork: bool) -> Self {
        self.fork_context = fork;
        self
    }

    /// Seed child state with parent messages and enable fork mode.
    #[must_use]
    pub fn with_fork_messages(mut self, messages: Vec<Message>) -> Self {
        self.fork_context = true;
        self.fork_messages = Some(messages);
        self
    }

    /// Replay a completed run from the host [`WorkflowRunStore`] (still charges agent budget).
    #[must_use]
    pub fn with_resume_from(mut self, id: impl Into<String>) -> Self {
        self.resume_from = Some(id.into());
        self
    }

    /// Attach parent event bus for spawn lifecycle on the parent stream.
    #[must_use]
    pub fn with_events(mut self, events: EventBus) -> Self {
        self.events = Some(events);
        self
    }
}

/// Result of a nested agent run.
#[derive(Debug, Clone)]
pub struct AgentRunResult {
    /// Child agent id.
    pub agent_id: AgentId,
    /// Optional label echoed from [`SpawnOpts`].
    pub label: Option<String>,
    /// Whether the run completed without runtime error.
    pub success: bool,
    /// Model text or structured payload.
    pub output: Value,
    /// Cancelled flag.
    pub cancelled: bool,
    /// Usage.
    pub usage: Usage,
    /// Wall duration ms.
    pub duration_ms: u64,
    /// Steps.
    pub steps: usize,
}

/// Host capable of spawning nested agents.
#[async_trait]
pub trait SessionHost: Send + Sync {
    /// Spawn and run a nested agent to completion.
    async fn spawn_agent(&self, opts: SpawnOpts) -> Result<AgentRunResult, OvoError>;

    /// Spawn many nested agents concurrently (order of results matches input).
    async fn spawn_agents(&self, opts: Vec<SpawnOpts>) -> Result<Vec<AgentRunResult>, OvoError> {
        try_join_all(opts.into_iter().map(|o| self.spawn_agent(o))).await
    }
}

/// Rebuilds the child tool pool from the isolation environment.
///
/// Production wiring lives on the facade (`sandboxed_host`); the kernel must
/// not compile `ovo-toolkit`. Factory failure is fail-closed (same path as
/// [`InProcessHost`] build errors: cleanup, refund, `SpawnFinished` false).
pub type ChildToolkit =
    Arc<dyn Fn(&IsolationEnv) -> Result<Vec<SharedTool>, OvoError> + Send + Sync>;

/// In-process host: nested [`TurnRuntime`] with shared sampler, tool pool, and limits.
///
/// [`InProcessHost::new`] is a test convenience: child agents clone `tools`
/// unless [`Self::with_child_toolkit`] is set; agent budget defaults to 128;
/// approval defaults to [`ovo_tools::AlwaysDeny`]. Passing `trusted_toolkit`
/// still means child processes are trusted.
///
/// There is no `InProcessHost::sandboxed`. Production wiring is facade
/// `sandboxed_host` (`feature = "toolkit"`).
pub struct InProcessHost {
    sampler: Arc<dyn LlmSampler>,
    tools: Vec<SharedTool>,
    base_instructions: String,
    runtime: TurnRuntime,
    /// Absolute cap on nested agent spawns (`None` = unlimited).
    agent_budget: Option<u64>,
    spent: AtomicU64,
    /// Max spawn depth (`None` = unlimited). Depth must satisfy `depth < max`.
    max_spawn_depth: Option<u32>,
    /// Concurrent in-flight children (`None` = unlimited).
    concurrency: Option<Arc<Semaphore>>,
    max_concurrent_children: Option<usize>,
    /// Named agent definitions for `agent_type` resolution.
    agent_registry: AgentRegistry,
    /// System prompt assembler (project AGENTS.md, etc.).
    prompt_assembler: Arc<dyn PromptAssembler>,
    /// Isolation backend for child environments (default in-process).
    isolation: Arc<dyn IsolationBackend>,
    metrics: SharedMetrics,
    /// Parent session handle: seeds `fork_context` when `fork_messages` is unset.
    parent_handle: Option<ChatStateHandle>,
    /// Workflow run store for `resume_from` lookups.
    run_store: Option<Arc<dyn WorkflowRunStore>>,
    approval: Arc<dyn ApprovalGate>,
    child_toolkit: Option<ChildToolkit>,
}

impl std::fmt::Debug for InProcessHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InProcessHost")
            .field("tools", &self.tools.len())
            .field("base_instructions_len", &self.base_instructions.len())
            .field("runtime", &self.runtime)
            .field("agent_budget", &self.agent_budget)
            .field("spent", &self.spent.load(Ordering::Relaxed))
            .field("max_spawn_depth", &self.max_spawn_depth)
            .field("max_concurrent_children", &self.max_concurrent_children)
            .field("agent_registry", &self.agent_registry.len())
            .field("isolation", &self.isolation.name())
            .field("has_approval", &true)
            .field("has_child_toolkit", &self.child_toolkit.is_some())
            .finish_non_exhaustive()
    }
}

impl InProcessHost {
    /// Create a host with default depth/concurrency caps, budget 128, and [`ovo_tools::AlwaysDeny`].
    ///
    /// Test convenience: children clone `tools` unless [`Self::with_child_toolkit`]
    /// is set. Production factory is facade `sandboxed_host`.
    #[must_use]
    pub fn new(sampler: Arc<dyn LlmSampler>, tools: Vec<SharedTool>) -> Self {
        Self {
            sampler,
            tools,
            base_instructions: "You are a focused sub-agent. Complete the task.".into(),
            runtime: TurnRuntime::new(),
            agent_budget: Some(ovo_workflow::DEFAULT_AGENT_BUDGET),
            spent: AtomicU64::new(0),
            max_spawn_depth: Some(DEFAULT_MAX_SPAWN_DEPTH),
            concurrency: Some(Arc::new(Semaphore::new(DEFAULT_MAX_CONCURRENT_CHILDREN))),
            max_concurrent_children: Some(DEFAULT_MAX_CONCURRENT_CHILDREN),
            agent_registry: AgentRegistry::with_builtins(),
            prompt_assembler: Arc::new(IdentityAssembler),
            isolation: Arc::new(InProcessIsolation),
            metrics: Arc::new(NoopMetrics),
            parent_handle: None,
            run_store: None,
            approval: Arc::new(ovo_tools::AlwaysDeny),
            child_toolkit: None,
        }
    }

    /// Absolute agent-call budget for this host (every successful admission counts 1).
    ///
    /// Values above [`ovo_workflow::MAX_AGENT_BUDGET`] are capped at 1024.
    #[must_use]
    pub fn with_agent_budget(mut self, budget: u64) -> Self {
        self.agent_budget = Some(budget.min(ovo_workflow::MAX_AGENT_BUDGET));
        self
    }

    /// Remove the host agent-budget cap. Requires [`crate::TrustedExecution`].
    #[must_use]
    pub fn with_unlimited_agent_budget(mut self, _t: crate::TrustedExecution) -> Self {
        self.agent_budget = None;
        self
    }

    /// Approval gate copied onto every child turn (`spawn_one`).
    #[must_use]
    pub fn with_approval(mut self, gate: Arc<dyn ApprovalGate>) -> Self {
        self.approval = gate;
        self
    }

    /// Rebuild child tools from isolation env instead of cloning the parent pool.
    #[must_use]
    pub fn with_child_toolkit(mut self, f: ChildToolkit) -> Self {
        self.child_toolkit = Some(f);
        self
    }

    /// Cap nesting depth (`depth` must be `< max`). `None` disables the limit.
    #[must_use]
    pub const fn with_max_spawn_depth(mut self, max: Option<u32>) -> Self {
        self.max_spawn_depth = max;
        self
    }

    /// Cap concurrent in-flight children. `None` disables the limit.
    #[must_use]
    pub fn with_max_concurrent_children(mut self, max: Option<usize>) -> Self {
        self.max_concurrent_children = max;
        self.concurrency = max.map(|n| Arc::new(Semaphore::new(n.max(1))));
        self
    }

    /// Install a full agent registry for `agent_type` resolution.
    #[must_use]
    pub fn with_agent_registry(mut self, registry: AgentRegistry) -> Self {
        self.agent_registry = registry;
        self
    }

    /// Register agent definitions (merged into the host registry).
    #[must_use]
    pub fn with_agent_definitions(
        mut self,
        defs: impl IntoIterator<Item = AgentDefinition>,
    ) -> Self {
        self.agent_registry = self
            .agent_registry
            .merge(&AgentRegistry::from_definitions(defs));
        self
    }

    /// Install a prompt assembler applied when resolving `agent_type` definitions.
    #[must_use]
    pub fn with_prompt_assembler(mut self, assembler: Arc<dyn PromptAssembler>) -> Self {
        self.prompt_assembler = assembler;
        self
    }

    /// Install an isolation backend (default [`InProcessIsolation`]).
    #[must_use]
    pub fn with_isolation(mut self, isolation: Arc<dyn IsolationBackend>) -> Self {
        self.isolation = isolation;
        self
    }

    /// Metrics sink for spawn/turn accounting.
    #[must_use]
    pub fn with_metrics(mut self, metrics: SharedMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    /// Parent conversation handle used when `fork_context` is set without messages.
    #[must_use]
    pub fn with_parent_handle(mut self, handle: ChatStateHandle) -> Self {
        self.parent_handle = Some(handle);
        self
    }

    /// Workflow run store enabling `resume_from` on spawn opts.
    #[must_use]
    pub fn with_run_store(mut self, store: Arc<dyn WorkflowRunStore>) -> Self {
        self.run_store = Some(store);
        self
    }

    /// Override child system instructions (used when `agent_type` is unset).
    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.base_instructions = instructions.into();
        self
    }

    /// Shared agent registry (clone is cheap).
    #[must_use]
    pub fn agent_registry(&self) -> &AgentRegistry {
        &self.agent_registry
    }

    /// Agents admitted so far (including in-flight after reservation).
    #[must_use]
    pub fn agents_spent(&self) -> u64 {
        self.spent.load(Ordering::Relaxed)
    }

    /// Remaining budget when capped.
    #[must_use]
    pub fn agents_remaining(&self) -> Option<u64> {
        self.agent_budget
            .map(|b| b.saturating_sub(self.spent.load(Ordering::Relaxed)))
    }

    /// Configured max spawn depth.
    #[must_use]
    pub const fn max_spawn_depth(&self) -> Option<u32> {
        self.max_spawn_depth
    }

    /// Configured max concurrent children.
    #[must_use]
    pub const fn max_concurrent_children(&self) -> Option<usize> {
        self.max_concurrent_children
    }

    fn check_depth(&self, depth: u32) -> Result<(), OvoError> {
        if let Some(max) = self.max_spawn_depth
            && depth >= max
        {
            return Err(OvoError::new(
                ErrorCode::HostDepth,
                format!("spawn depth {depth} exceeds max_spawn_depth {max}"),
            ));
        }
        Ok(())
    }

    fn check_fork_opts(opts: &SpawnOpts, has_parent_handle: bool) -> Result<(), OvoError> {
        if opts.fork_context && opts.fork_messages.is_none() && !has_parent_handle {
            return Err(OvoError::new(
                ErrorCode::HostUnsupported,
                "fork_context requires fork_messages or host parent_handle",
            ));
        }
        Ok(())
    }

    async fn child_state(&self, opts: &SpawnOpts) -> Result<VecConversationState, OvoError> {
        if !opts.fork_context {
            return Ok(VecConversationState::new());
        }
        if let Some(msgs) = &opts.fork_messages {
            return Ok(VecConversationState::from_messages(msgs.clone()));
        }
        if let Some(handle) = &self.parent_handle {
            let msgs = handle.messages().await?;
            return Ok(VecConversationState::from_messages(msgs));
        }
        Err(OvoError::new(
            ErrorCode::HostUnsupported,
            "fork_context requires fork_messages or host parent_handle",
        ))
    }

    /// Resolve `resume_from` via [`WorkflowRunStore`] when configured.
    fn try_resume(&self, opts: &SpawnOpts) -> Result<Option<AgentRunResult>, OvoError> {
        let Some(id) = opts.resume_from.as_deref() else {
            return Ok(None);
        };
        let Some(store) = &self.run_store else {
            return Err(OvoError::new(
                ErrorCode::HostUnsupported,
                "resume_from requires host run_store (WorkflowRunStore)",
            ));
        };
        let rec = store
            .get(id)
            .map_err(|e| OvoError::new(ErrorCode::HostSpawn, format!("workflow run store: {e}")))?;
        let Some(rec) = rec else {
            return Err(OvoError::new(
                ErrorCode::HostUnsupported,
                format!("resume_from run_id '{id}' not found in WorkflowRunStore"),
            ));
        };
        match rec.status {
            WorkflowRunStatus::Completed => {
                let output = rec.result.clone().unwrap_or(Value::Null);
                Ok(Some(AgentRunResult {
                    agent_id: AgentId::generate(),
                    label: opts.label.clone().or_else(|| Some(rec.name.clone())),
                    success: true,
                    output,
                    cancelled: false,
                    usage: Usage::zero(),
                    duration_ms: 0,
                    steps: 0,
                }))
            }
            WorkflowRunStatus::Paused | WorkflowRunStatus::BudgetExceeded => Err(OvoError::new(
                ErrorCode::HostUnsupported,
                format!(
                    "resume_from '{id}' is {:?}; resume via workflow engine (journal {})",
                    rec.status,
                    rec.journal_path.display()
                ),
            )),
            other => Err(OvoError::new(
                ErrorCode::HostUnsupported,
                format!("resume_from '{id}' has non-resumable status {other:?}"),
            )),
        }
    }

    fn try_acquire_concurrency(&self) -> Result<Option<OwnedSemaphorePermit>, OvoError> {
        let Some(sem) = &self.concurrency else {
            return Ok(None);
        };
        match Arc::clone(sem).try_acquire_owned() {
            Ok(permit) => Ok(Some(permit)),
            Err(_) => Err(OvoError::new(
                ErrorCode::HostConcurrency,
                format!(
                    "max concurrent children reached ({})",
                    self.max_concurrent_children.unwrap_or(0)
                ),
            )),
        }
    }

    fn reserve_slot(&self) -> Result<(), OvoError> {
        let Some(budget) = self.agent_budget else {
            self.spent.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        };
        self.reserve_against_budget(budget)
    }

    /// Refund a slot reserved before the child turn actually starts.
    fn release_slot(&self) {
        self.spent
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |s| {
                Some(s.saturating_sub(1))
            })
            .ok();
    }

    fn reserve_against_budget(&self, budget: u64) -> Result<(), OvoError> {
        loop {
            let spent = self.spent.load(Ordering::Acquire);
            if spent >= budget {
                return Err(OvoError::new(
                    ErrorCode::HostBudget,
                    format!("agent budget exhausted: spent {spent}, maximum {budget}"),
                ));
            }
            if self
                .spent
                .compare_exchange(spent, spent + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    fn effective_capability(&self, opts: &SpawnOpts) -> CapabilityMode {
        let mut mode = opts.capability_mode;
        if let Some(name) = opts.agent_type.as_deref()
            && let Some(def) = self.agent_registry.get(name)
            && let Some(def_cap) = def.capability
        {
            mode = mode.intersect(def_cap);
        }
        mode
    }

    fn build_child(
        &self,
        opts: &SpawnOpts,
        isolation_env: &IsolationEnv,
    ) -> Result<Agent, OvoError> {
        let tools = if let Some(f) = &self.child_toolkit {
            f(isolation_env)?
        } else {
            self.tools.clone()
        };
        let mut builder = if let Some(name) = opts.agent_type.as_deref() {
            let def = self.agent_registry.require(name)?.clone();
            let system = self.prompt_assembler.assemble(&def)?;
            AgentBuilder::from_definition(def)
                .instructions(system)
                .tools(tools)
        } else {
            let name = opts.label.clone().unwrap_or_else(|| "subagent".to_owned());
            AgentBuilder::named(name)
                .instructions(self.base_instructions.clone())
                .tools(tools)
        };

        if let Some(model) = &opts.model {
            builder = builder.model(model.clone());
        }
        if let Some(max_steps) = opts.max_steps {
            builder = builder.max_steps(max_steps);
        }
        if let Some(schema) = opts.output_schema.clone() {
            builder = builder.output_schema(schema);
        }
        builder.build()
    }

    #[allow(
        clippy::too_many_lines,
        clippy::excessive_nesting,
        reason = "spawn_one owns budget/isolation/turn/events end-to-end"
    )]
    async fn spawn_one(&self, opts: SpawnOpts) -> Result<AgentRunResult, OvoError> {
        let agent_id = AgentId::generate();
        let label = opts.label.clone();
        let parent = Span::current();
        let span = info_span!(
            parent: parent,
            "ovo.spawn",
            ovo.agent_id = %agent_id,
            ovo.agent_label = label.as_deref().unwrap_or(""),
            ovo.agent_type = opts.agent_type.as_deref().unwrap_or(""),
            ovo.capability = ?opts.capability_mode,
            ovo.spawn_depth = opts.depth,
        );

        async move {
            if opts.cancel.is_cancelled() {
                return Err(OvoError::new(
                    ErrorCode::HostCancelled,
                    "spawn cancelled before start",
                ));
            }
            Self::check_fork_opts(&opts, self.parent_handle.is_some())?;
            self.check_depth(opts.depth)?;
            // Budget + concurrency apply to resume_from (no free spawn path).
            let _permit = self.try_acquire_concurrency()?;
            self.reserve_slot()?;
            match self.try_resume(&opts) {
                Ok(Some(resumed)) => {
                    record_spawn(self.metrics.as_ref(), "ok");
                    return Ok(resumed);
                }
                Ok(None) => {}
                Err(e) => {
                    self.release_slot();
                    return Err(e);
                }
            }

            if let Some(bus) = &opts.events {
                bus.emit(
                    None,
                    Some(opts.depth),
                    TurnEventKind::SpawnStarted {
                        child_agent_id: agent_id.to_string(),
                        label: label.clone(),
                        depth: opts.depth,
                    },
                );
            }
            let emit_spawn_failed = |bus: &EventBus| {
                bus.emit(
                    None,
                    Some(opts.depth),
                    TurnEventKind::SpawnFinished {
                        child_agent_id: agent_id.to_string(),
                        label: label.clone(),
                        depth: opts.depth,
                        success: false,
                        cancelled: false,
                    },
                );
            };

            let isolation_env = match self.isolation.prepare(&opts).await {
                Ok(env) => env,
                Err(e) => {
                    self.release_slot();
                    if let Some(bus) = &opts.events {
                        emit_spawn_failed(bus);
                    }
                    return Err(e);
                }
            };
            let started = Instant::now();
            let agent = match self.build_child(&opts, &isolation_env) {
                Ok(a) => a,
                Err(e) => {
                    let _ = self.isolation.cleanup(&isolation_env).await;
                    self.release_slot();
                    if let Some(bus) = &opts.events {
                        emit_spawn_failed(bus);
                    }
                    return Err(e);
                }
            };
            let mut state = match self.child_state(&opts).await {
                Ok(s) => s,
                Err(e) => {
                    let _ = self.isolation.cleanup(&isolation_env).await;
                    self.release_slot();
                    if let Some(bus) = &opts.events {
                        emit_spawn_failed(bus);
                    }
                    return Err(e);
                }
            };
            let capability_mode = self.effective_capability(&opts);
            let max_steps = opts.max_steps.or_else(|| {
                opts.agent_type
                    .as_deref()
                    .and_then(|n| self.agent_registry.get(n).map(|d| d.max_steps))
            });
            let max_output_tokens = opts.max_output_tokens.and_then(|n| u32::try_from(n).ok());
            // Child turn shares parent bus so seq stays monotonic on one stream.
            let turn_opts = TurnOptions {
                max_steps,
                capability_mode,
                cancel: opts.cancel.clone(),
                agent_id: Some(agent_id.clone()),
                metrics: Arc::clone(&self.metrics),
                spawn_depth: Some(opts.depth),
                max_output_tokens,
                cwd: isolation_env.cwd.clone(),
                events: opts.events.clone(),
                approval: Arc::clone(&self.approval),
                approval_policy: ApprovalPolicy::Destructive,
                ..TurnOptions::default()
            };
            // Once the turn starts, the slot is consumed even on error.
            let outcome = match self
                .runtime
                .run(
                    &agent,
                    self.sampler.as_ref(),
                    &mut state,
                    TurnInput::Text(opts.prompt),
                    turn_opts,
                )
                .await
            {
                Ok(o) => o,
                Err(e) => {
                    let _ = self.isolation.cleanup(&isolation_env).await;
                    record_spawn(self.metrics.as_ref(), "error");
                    if let Some(bus) = &opts.events {
                        bus.emit(
                            None,
                            Some(opts.depth),
                            TurnEventKind::SpawnFinished {
                                child_agent_id: agent_id.to_string(),
                                label: label.clone(),
                                depth: opts.depth,
                                success: false,
                                cancelled: false,
                            },
                        );
                    }
                    return Err(map_turn_error(e));
                }
            };
            // Cleanup failure must not invert a successful turn (parent would lose the result).
            if let Err(_e) = self.isolation.cleanup(&isolation_env).await {
                record_spawn(self.metrics.as_ref(), "cleanup_error");
            }
            let status = if outcome.cancelled { "cancelled" } else { "ok" };
            record_spawn(self.metrics.as_ref(), status);

            if let Some(bus) = &opts.events {
                bus.emit(
                    None,
                    Some(opts.depth),
                    TurnEventKind::SpawnFinished {
                        child_agent_id: agent_id.to_string(),
                        label: label.clone(),
                        depth: opts.depth,
                        success: !outcome.cancelled,
                        cancelled: outcome.cancelled,
                    },
                );
            }

            let output = outcome
                .output_json
                .unwrap_or_else(|| Value::String(outcome.output_text.clone()));
            Ok(AgentRunResult {
                agent_id,
                label,
                success: !outcome.cancelled,
                output,
                cancelled: outcome.cancelled,
                usage: outcome.usage,
                duration_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
                steps: outcome.steps,
            })
        }
        .instrument(span)
        .await
    }
}

fn map_turn_error(e: OvoError) -> OvoError {
    if matches!(
        e.code(),
        ErrorCode::RuntimeCancelled | ErrorCode::LlmCancelled
    ) {
        OvoError::new(ErrorCode::HostCancelled, e.message().to_owned()).with_source(e)
    } else {
        OvoError::new(ErrorCode::HostSpawn, e.message().to_owned()).with_source(e)
    }
}

#[async_trait]
impl SessionHost for InProcessHost {
    async fn spawn_agent(&self, opts: SpawnOpts) -> Result<AgentRunResult, OvoError> {
        self.spawn_one(opts).await
    }

    async fn spawn_agents(&self, opts: Vec<SpawnOpts>) -> Result<Vec<AgentRunResult>, OvoError> {
        try_join_all(opts.into_iter().map(|o| self.spawn_one(o))).await
    }
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::excessive_nesting,
    reason = "unit tests use expect and nested mock structs"
)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use ovo_llm::MockSampler;
    use ovo_tools::{DynTool, ToolCallContext, ToolMetadata, ToolResult};
    use ovo_types::{ErrorCode, Message, ToolCall, ToolCallId};
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn concurrent_two_workers() {
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("task a", "worker-a-result");
        sampler.map_user_text("task b", "worker-b-result");
        let host = InProcessHost::new(sampler, vec![]);
        let results = host
            .spawn_agents(vec![
                SpawnOpts::new("task a").with_label("alpha"),
                SpawnOpts::new("task b").with_label("beta"),
            ])
            .await
            .expect("spawn");
        assert_eq!(results.len(), 2);
        assert_eq!(
            results.first().and_then(|r| r.label.as_deref()),
            Some("alpha")
        );
        assert_eq!(
            results.get(1).and_then(|r| r.label.as_deref()),
            Some("beta")
        );
        assert_eq!(
            results.first().map(|r| &r.output),
            Some(&Value::String("worker-a-result".into()))
        );
        assert_eq!(
            results.get(1).map(|r| &r.output),
            Some(&Value::String("worker-b-result".into()))
        );
        assert_eq!(host.agents_spent(), 2);
    }

    #[tokio::test]
    async fn budget_exhausted() {
        let sampler = Arc::new(MockSampler::new());
        sampler.push_text("only-one");
        let host = InProcessHost::new(sampler, vec![]).with_agent_budget(1);
        host.spawn_agent(SpawnOpts::new("first"))
            .await
            .expect("first");
        let err = host
            .spawn_agent(SpawnOpts::new("second"))
            .await
            .expect_err("budget");
        assert_eq!(err.code(), ErrorCode::HostBudget);
    }

    #[tokio::test]
    async fn cancel_before_start() {
        let sampler = Arc::new(MockSampler::new());
        let host = InProcessHost::new(sampler, vec![]);
        let cancel = CancellationToken::new();
        cancel.cancel();
        let err = host
            .spawn_agent(SpawnOpts::new("x").with_cancel(cancel))
            .await
            .expect_err("cancel");
        assert_eq!(err.code(), ErrorCode::HostCancelled);
    }

    #[tokio::test]
    async fn depth_fail_closed() {
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("ok", "done");
        let host = InProcessHost::new(sampler, vec![]).with_max_spawn_depth(Some(1));
        host.spawn_agent(SpawnOpts::new("ok").with_depth(0))
            .await
            .expect("depth 0");
        let err = host
            .spawn_agent(SpawnOpts::new("ok").with_depth(1))
            .await
            .expect_err("depth");
        assert_eq!(err.code(), ErrorCode::HostDepth);
    }

    #[tokio::test]
    async fn concurrency_fail_closed() {
        struct HoldingSampler {
            inner: MockSampler,
            release: tokio::sync::Notify,
            entered: tokio::sync::Notify,
        }

        #[async_trait]
        impl LlmSampler for HoldingSampler {
            async fn sample(
                &self,
                request: ovo_llm::SampleRequest,
            ) -> Result<ovo_llm::SampleResponse, OvoError> {
                self.entered.notify_one();
                self.release.notified().await;
                self.inner.sample(request).await
            }
        }

        let holder = Arc::new(HoldingSampler {
            inner: MockSampler::new(),
            release: tokio::sync::Notify::new(),
            entered: tokio::sync::Notify::new(),
        });
        holder.inner.map_user_text("slow", "done");
        holder.inner.map_user_text("fast", "nope");

        let sampler: Arc<dyn LlmSampler> = holder.clone();
        let host =
            Arc::new(InProcessHost::new(sampler, vec![]).with_max_concurrent_children(Some(1)));

        let h1 = Arc::clone(&host);
        let t1 = tokio::spawn(async move { h1.spawn_agent(SpawnOpts::new("slow")).await });
        holder.entered.notified().await;

        let err = host
            .spawn_agent(SpawnOpts::new("fast"))
            .await
            .expect_err("concurrency");
        assert_eq!(err.code(), ErrorCode::HostConcurrency);

        holder.release.notify_one();
        t1.await.expect("join").expect("first ok");
    }

    #[tokio::test]
    async fn fork_context_requires_messages() {
        let sampler = Arc::new(MockSampler::new());
        let host = InProcessHost::new(sampler, vec![]);
        let err = host
            .spawn_agent(SpawnOpts::new("x").with_fork_context(true))
            .await
            .expect_err("fork");
        assert_eq!(err.code(), ErrorCode::HostUnsupported);
    }

    #[tokio::test]
    async fn fork_context_from_parent_handle() {
        use ovo_state::ChatStateHandle;
        use ovo_types::Message;

        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("continue", "handle-fork");
        let handle = ChatStateHandle::spawn(vec![Message::user("seed"), Message::assistant("a")]);
        let host = InProcessHost::new(sampler, vec![]).with_parent_handle(handle);
        let run = host
            .spawn_agent(SpawnOpts::new("continue").with_fork_context(true))
            .await
            .expect("fork");
        assert_eq!(run.output, Value::String("handle-fork".into()));
    }

    #[tokio::test]
    async fn resume_from_completed_run_store() {
        use std::path::PathBuf;
        use std::sync::Arc;

        use ovo_workflow::{MemoryWorkflowRunStore, WorkflowOutcome, WorkflowRunRecord};

        let sampler = Arc::new(MockSampler::new());
        let store = Arc::new(MemoryWorkflowRunStore::new());
        let mut rec = WorkflowRunRecord::new_running("r1", "wf", PathBuf::from("/tmp/j.jsonl"));
        rec.apply_outcome(&WorkflowOutcome::Completed {
            result: json!({"ok": true, "v": 1}),
        });
        store.put(rec).expect("put");
        let host = InProcessHost::new(sampler, vec![])
            .with_agent_budget(2)
            .with_run_store(store);
        let run = host
            .spawn_agent(SpawnOpts::new("unused").with_resume_from("r1"))
            .await
            .expect("resume");
        assert!(run.success);
        assert_eq!(run.output, json!({"ok": true, "v": 1}));
        assert_eq!(host.agents_spent(), 1, "resume_from charges budget");
    }

    #[tokio::test]
    async fn builtin_explore_spawnable() {
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("look", "found");
        let host = InProcessHost::new(sampler, vec![]);
        let run = host
            .spawn_agent(SpawnOpts::new("look").with_agent_type("explore"))
            .await
            .expect("explore");
        assert_eq!(run.output, Value::String("found".into()));
    }

    #[tokio::test]
    async fn fork_context_seeds_parent_messages() {
        use ovo_types::Message;

        let sampler = Arc::new(MockSampler::new());
        // Child user prompt is "continue"; parent context is already in state.
        sampler.map_user_text("continue", "forked-ok");
        let host = InProcessHost::new(sampler, vec![]);
        let parent = vec![
            Message::system("parent-sys"),
            Message::user("earlier"),
            Message::assistant("prior answer"),
        ];
        let run = host
            .spawn_agent(SpawnOpts::new("continue").with_fork_messages(parent))
            .await
            .expect("fork spawn");
        assert_eq!(run.output, Value::String("forked-ok".into()));
    }

    #[tokio::test]
    async fn agent_type_not_found() {
        let sampler = Arc::new(MockSampler::new());
        let host = InProcessHost::new(sampler, vec![]);
        let err = host
            .spawn_agent(SpawnOpts::new("x").with_agent_type("missing"))
            .await
            .expect_err("type");
        assert_eq!(err.code(), ErrorCode::AgentNotFound);
    }

    #[tokio::test]
    async fn agent_type_resolves_definition() {
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("do work", "from-def");
        let mut def = AgentDefinition::new("worker");
        def.description = "w".into();
        def.instructions = ovo_agent::Instructions::Static("Be brief.".into());
        def.model = "mock".into();
        def.max_steps = 4;
        let reg = AgentRegistry::from_definitions([def]);
        let host = InProcessHost::new(sampler, vec![]).with_agent_registry(reg);
        let run = host
            .spawn_agent(SpawnOpts::new("do work").with_agent_type("worker"))
            .await
            .expect("spawn");
        assert_eq!(run.output, Value::String("from-def".into()));
    }

    #[tokio::test]
    async fn prompt_assembler_applied_for_agent_type() {
        use ovo_agent::ProjectPromptAssembler;

        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("task", "done");
        let mut def = AgentDefinition::new("worker");
        def.description = "w".into();
        def.instructions = ovo_agent::Instructions::Static("Body.".into());
        def.model = "mock".into();
        def.max_steps = 4;
        let asm = Arc::new(ProjectPromptAssembler::with_preamble("PREAMBLE_MARK"));
        let host = InProcessHost::new(sampler, vec![])
            .with_agent_definitions([def])
            .with_prompt_assembler(asm);
        let run = host
            .spawn_agent(SpawnOpts::new("task").with_agent_type("worker"))
            .await
            .expect("spawn");
        assert!(run.success);
        // Assembler applied at build time; spawn still succeeds with mock.
        assert_eq!(run.output, Value::String("done".into()));
    }

    #[tokio::test]
    async fn output_schema_field_accepted() {
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("schema", r#"{"ok":true}"#);
        let host = InProcessHost::new(sampler, vec![]);
        let schema = json!({
            "type": "object",
            "properties": { "ok": { "type": "boolean" } },
            "required": ["ok"]
        });
        let run = host
            .spawn_agent(SpawnOpts::new("schema").with_output_schema(schema))
            .await
            .expect("spawn");
        assert!(run.success);
    }

    #[tokio::test]
    async fn isolation_prepare_cleanup_on_spawn() {
        use std::sync::atomic::{AtomicUsize, Ordering as AtOrd};

        use crate::isolation::{IsolationBackend, IsolationEnv};

        struct CountingIsolation {
            prepares: Arc<AtomicUsize>,
            cleanups: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl IsolationBackend for CountingIsolation {
            fn name(&self) -> &'static str {
                "counting"
            }

            async fn prepare(&self, opts: &SpawnOpts) -> Result<IsolationEnv, OvoError> {
                self.prepares.fetch_add(1, AtOrd::SeqCst);
                Ok(IsolationEnv {
                    cwd: None,
                    label: opts.label.clone(),
                })
            }

            async fn cleanup(&self, _env: &IsolationEnv) -> Result<(), OvoError> {
                self.cleanups.fetch_add(1, AtOrd::SeqCst);
                Ok(())
            }
        }

        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("iso", "ok");
        let prepares = Arc::new(AtomicUsize::new(0));
        let cleanups = Arc::new(AtomicUsize::new(0));
        let host =
            InProcessHost::new(sampler, vec![]).with_isolation(Arc::new(CountingIsolation {
                prepares: Arc::clone(&prepares),
                cleanups: Arc::clone(&cleanups),
            }));
        host.spawn_agent(SpawnOpts::new("iso").with_label("child"))
            .await
            .expect("spawn");
        assert_eq!(prepares.load(AtOrd::SeqCst), 1, "prepare once");
        assert_eq!(cleanups.load(AtOrd::SeqCst), 1, "cleanup once");
    }

    #[tokio::test]
    async fn isolation_prepare_fail_closed() {
        use crate::isolation::{IsolationBackend, IsolationEnv, isolation_error};

        struct FailPrepare;

        #[async_trait]
        impl IsolationBackend for FailPrepare {
            fn name(&self) -> &'static str {
                "fail"
            }

            async fn prepare(&self, _opts: &SpawnOpts) -> Result<IsolationEnv, OvoError> {
                Err(isolation_error(self.name(), "no sandbox available"))
            }

            async fn cleanup(&self, _env: &IsolationEnv) -> Result<(), OvoError> {
                Ok(())
            }
        }

        let sampler = Arc::new(MockSampler::new());
        let host = InProcessHost::new(sampler, vec![]).with_isolation(Arc::new(FailPrepare));
        let err = host
            .spawn_agent(SpawnOpts::new("x"))
            .await
            .expect_err("iso fail");
        assert_eq!(err.code(), ErrorCode::HostIsolation);
        assert_eq!(
            host.agents_spent(),
            0,
            "pre-start isolation failure refunds budget"
        );
    }

    #[test]
    fn new_default_budget_is_128() {
        let sampler = Arc::new(MockSampler::new());
        let host = InProcessHost::new(sampler, vec![]);
        assert_eq!(host.agents_remaining(), Some(128));
    }

    #[test]
    fn with_agent_budget_caps_at_1024() {
        let sampler = Arc::new(MockSampler::new());
        let host = InProcessHost::new(sampler, vec![]).with_agent_budget(u64::MAX);
        assert_eq!(host.agents_remaining(), Some(1024));
    }

    #[test]
    fn unlimited_requires_trusted() {
        let sampler = Arc::new(MockSampler::new());
        let host = InProcessHost::new(sampler, vec![])
            .with_unlimited_agent_budget(crate::TrustedExecution);
        assert_eq!(host.agents_remaining(), None);
    }

    struct WriteStub {
        called: Arc<AtomicBool>,
    }

    #[async_trait]
    impl DynTool for WriteStub {
        fn name(&self) -> &'static str {
            "write_stub"
        }
        fn description(&self) -> &'static str {
            "write"
        }
        fn parameters(&self) -> Value {
            json!({"type":"object","properties":{}})
        }
        fn metadata(&self) -> ToolMetadata {
            ToolMetadata::exclusive_write()
        }
        async fn call(
            &self,
            _ctx: ToolCallContext,
            _arguments: Value,
        ) -> Result<ToolResult, OvoError> {
            self.called.store(true, Ordering::SeqCst);
            Ok(ToolResult::text("wrote"))
        }
    }

    struct InspectingSampler {
        inner: MockSampler,
        saw_denied: Arc<AtomicBool>,
    }

    #[async_trait]
    impl LlmSampler for InspectingSampler {
        async fn sample(
            &self,
            request: ovo_llm::SampleRequest,
        ) -> Result<ovo_llm::SampleResponse, OvoError> {
            let texts: String = request.messages.iter().map(Message::text).collect();
            if texts.contains("approval denied") {
                self.saw_denied.store(true, Ordering::SeqCst);
            }
            self.inner.sample(request).await
        }
    }

    #[tokio::test]
    async fn spawn_one_uses_host_always_deny() {
        let called = Arc::new(AtomicBool::new(false));
        let saw_denied = Arc::new(AtomicBool::new(false));
        let inner = MockSampler::new();
        let id = ToolCallId::new("w1").expect("id");
        inner.push_tools(Message::assistant_tools(vec![ToolCall {
            id,
            name: "write_stub".into(),
            arguments: json!({}),
        }]));
        inner.push_text("after-deny");
        let sampler = Arc::new(InspectingSampler {
            inner,
            saw_denied: Arc::clone(&saw_denied),
        });
        let host = InProcessHost::new(
            sampler,
            vec![Arc::new(WriteStub {
                called: Arc::clone(&called),
            })],
        );
        let run = host
            .spawn_agent(SpawnOpts::new("write"))
            .await
            .expect("spawn");
        assert!(run.success);
        assert!(
            !called.load(Ordering::SeqCst),
            "AlwaysDeny must block write"
        );
        assert!(
            saw_denied.load(Ordering::SeqCst),
            "child tool result must contain approval denied"
        );
    }

    struct NamedTool {
        name: &'static str,
        hits: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl DynTool for NamedTool {
        fn name(&self) -> &'static str {
            self.name
        }
        fn description(&self) -> &'static str {
            self.name
        }
        fn parameters(&self) -> Value {
            json!({"type":"object","properties":{}})
        }
        fn metadata(&self) -> ToolMetadata {
            ToolMetadata::read_only()
        }
        async fn call(
            &self,
            _ctx: ToolCallContext,
            _arguments: Value,
        ) -> Result<ToolResult, OvoError> {
            self.hits.fetch_add(1, Ordering::SeqCst);
            Ok(ToolResult::text(self.name))
        }
    }

    #[tokio::test]
    async fn child_toolkit_rebuilds() {
        let parent_hits = Arc::new(AtomicUsize::new(0));
        let child_hits = Arc::new(AtomicUsize::new(0));
        let factory_calls = Arc::new(AtomicUsize::new(0));
        let sampler = Arc::new(MockSampler::new());
        let id = ToolCallId::new("c1").expect("id");
        sampler.push_tools(Message::assistant_tools(vec![ToolCall {
            id,
            name: "child_only".into(),
            arguments: json!({}),
        }]));
        sampler.push_text("child-ok");

        let child_hits_f = Arc::clone(&child_hits);
        let factory_calls_f = Arc::clone(&factory_calls);
        let host = InProcessHost::new(
            sampler,
            vec![Arc::new(NamedTool {
                name: "parent_only",
                hits: Arc::clone(&parent_hits),
            })],
        )
        .with_child_toolkit(Arc::new(move |_env: &IsolationEnv| {
            factory_calls_f.fetch_add(1, Ordering::SeqCst);
            Ok(vec![Arc::new(NamedTool {
                name: "child_only",
                hits: Arc::clone(&child_hits_f),
            })])
        }));

        let run = host
            .spawn_agent(SpawnOpts::new("task"))
            .await
            .expect("spawn");
        assert_eq!(run.output, Value::String("child-ok".into()));
        assert_eq!(factory_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            parent_hits.load(Ordering::SeqCst),
            0,
            "parent tools must not be cloned into the child"
        );
        assert_eq!(
            child_hits.load(Ordering::SeqCst),
            1,
            "child toolkit tools must run"
        );
    }
}
