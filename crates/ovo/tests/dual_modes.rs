//! Dual multi-agent contract suite: Mode A (dynamic) + Mode B (journaled workflow).
#![allow(
    unused_crate_dependencies,
    reason = "integration binary links facade feature deps"
)]

#[cfg(test)]
mod dual {
    #![allow(
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::missing_assert_message,
        clippy::panic,
        clippy::excessive_nesting,
        reason = "integration tests use expect/panic for setup and assert outcomes"
    )]

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use ovo::{
        ChatStateHandle, DynTool, ErrorCode, HostError, InProcessHost, Journal, JsonlPersistence,
        LlmSampler, Message, MockSampler, OvoError, SampleRequest, SampleResponse, SessionHost,
        SpawnAgentTool, SpawnOpts, ToolCallContext, Usage, WorkflowOutcome, WorkflowRunParams,
        run_workflow_on_host,
    };
    use serde_json::json;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    // ── Mode A ──────────────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_dynamic_delegation_two_workers() {
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("do A", "result-A");
        sampler.map_user_text("do B", "result-B");

        let host = InProcessHost::new(sampler, Vec::new()).with_agent_budget(8);
        let results = host
            .spawn_agents(vec![
                SpawnOpts::new("do A").with_label("A"),
                SpawnOpts::new("do B").with_label("B"),
            ])
            .await
            .expect("spawn_agents");

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].label.as_deref(), Some("A"));
        assert_eq!(results[1].label.as_deref(), Some("B"));
        assert_eq!(results[0].output.as_str(), Some("result-A"));
        assert_eq!(results[1].output.as_str(), Some("result-B"));
        assert!(results.iter().all(|r| r.success && !r.cancelled));
        assert_eq!(host.agents_spent(), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_budget_fail_closed() {
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("only", "ok");
        let host = InProcessHost::new(sampler, Vec::new()).with_agent_budget(1);
        host.spawn_agent(SpawnOpts::new("only"))
            .await
            .expect("first ok");
        let err = host
            .spawn_agent(SpawnOpts::new("second"))
            .await
            .expect_err("budget");
        assert_eq!(err.code(), ErrorCode::HostBudget);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cancel_before_start() {
        let sampler = Arc::new(MockSampler::new());
        let host = InProcessHost::new(sampler, Vec::new());
        let cancel = CancellationToken::new();
        cancel.cancel();
        let err = host
            .spawn_agent(SpawnOpts::new("x").with_cancel(cancel))
            .await
            .expect_err("cancel");
        assert_eq!(err.code(), ErrorCode::HostCancelled);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_depth_exceeded() {
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("ok", "done");
        let host = InProcessHost::new(sampler, Vec::new()).with_max_spawn_depth(Some(1));
        host.spawn_agent(SpawnOpts::new("ok").with_depth(0))
            .await
            .expect("depth 0");
        let err = host
            .spawn_agent(SpawnOpts::new("ok").with_depth(1))
            .await
            .expect_err("depth");
        assert_eq!(err.code(), ErrorCode::HostDepth);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_concurrent_children_cap() {
        struct HoldingSampler {
            inner: MockSampler,
            release: tokio::sync::Notify,
            entered: tokio::sync::Notify,
        }

        #[async_trait]
        impl LlmSampler for HoldingSampler {
            async fn sample(&self, request: SampleRequest) -> Result<SampleResponse, OvoError> {
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
        holder.inner.map_user_text("fast", "x");

        let sampler: Arc<dyn LlmSampler> = holder.clone();
        let host =
            Arc::new(InProcessHost::new(sampler, Vec::new()).with_max_concurrent_children(Some(1)));
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_spawn_agent_tool_e2e() {
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("child task", "child-done");
        let host: Arc<dyn SessionHost> = Arc::new(InProcessHost::new(sampler, Vec::new()));
        let tool = SpawnAgentTool::new(host);
        let result = tool
            .call(
                ToolCallContext::default(),
                json!({"prompt": "child task", "label": "w1"}),
            )
            .await
            .expect("call");
        assert!(!result.is_error);
        assert_eq!(
            result
                .structured
                .as_ref()
                .and_then(|s| s.get("output"))
                .and_then(|v| v.as_str()),
            Some("child-done")
        );
    }

    // ── Mode B ──────────────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn b_workflow_plan_and_parallel() {
        let sampler = Arc::new(MockSampler::new());
        sampler.push_text("plan-out");
        sampler.map_user_text("w0", "w0-out");
        sampler.map_user_text("w1", "w1-out");
        let host: Arc<dyn SessionHost> = Arc::new(InProcessHost::new(sampler, Vec::new()));

        let script = r#"
            let meta = #{ name: "it-fanout", description: "integration" };
            let plan = agent("plan", #{ label: "planner" });
            let ws = parallel([
                #{ prompt: "w0", label: "w0" },
                #{ prompt: "w1", label: "w1" },
            ]);
            complete(#{ plan: plan, workers: ws });
        "#;
        let (tx, _rx) = mpsc::unbounded_channel();
        let outcome = run_workflow_on_host(
            host,
            WorkflowRunParams {
                script: script.into(),
                args: json!({}),
                journal: Journal::new(None),
                host_tx: tx,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            },
            Some(16),
        )
        .await
        .expect("workflow");

        let WorkflowOutcome::Completed { result } = outcome else {
            panic!("expected completed: {outcome:?}");
        };
        assert_eq!(
            result.pointer("/plan/output").and_then(|v| v.as_str()),
            Some("plan-out")
        );
        let workers = result
            .get("workers")
            .and_then(|v| v.as_array())
            .expect("workers");
        assert_eq!(workers.len(), 2);
        assert_eq!(
            workers[0].get("output").and_then(|v| v.as_str()),
            Some("w0-out")
        );
        assert_eq!(
            workers[1].get("output").and_then(|v| v.as_str()),
            Some("w1-out")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn b_resume_skips_completed_host_calls() {
        struct CountMock {
            inner: MockSampler,
            calls: AtomicUsize,
        }

        #[async_trait]
        impl LlmSampler for CountMock {
            async fn sample(&self, request: SampleRequest) -> Result<SampleResponse, OvoError> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                self.inner.sample(request).await
            }
        }

        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("j.jsonl");
        let sampler = Arc::new(CountMock {
            inner: MockSampler::new(),
            calls: AtomicUsize::new(0),
        });
        sampler.inner.push_text("first");
        sampler.inner.push_text("second");

        let host: Arc<dyn SessionHost> = Arc::new(InProcessHost::new(sampler.clone(), Vec::new()));
        let script = r#"
            let meta = #{ name: "it-resume", description: "integration" };
            let a = agent("a");
            let b = agent("b");
            complete(#{ a: a, b: b });
        "#;

        let (tx, _rx) = mpsc::unbounded_channel();
        let o1 = run_workflow_on_host(
            Arc::clone(&host),
            WorkflowRunParams {
                script: script.into(),
                args: json!({}),
                journal: Journal::new(Some(path.clone())),
                host_tx: tx,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            },
            Some(8),
        )
        .await
        .expect("run1");
        assert!(matches!(o1, WorkflowOutcome::Completed { .. }));
        let after_first = sampler.calls.load(Ordering::SeqCst);
        assert_eq!(after_first, 2, "first run samples twice");

        // Cross-process simulation: drop host path and reload journal from disk.
        let journal = Journal::load(path).expect("load");
        assert_eq!(journal.len(), 2);
        let (tx2, _rx2) = mpsc::unbounded_channel();
        let o2 = run_workflow_on_host(
            host,
            WorkflowRunParams {
                script: script.into(),
                args: json!({}),
                journal,
                host_tx: tx2,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            },
            Some(8),
        )
        .await
        .expect("run2");
        assert!(matches!(o2, WorkflowOutcome::Completed { .. }));
        assert_eq!(
            sampler.calls.load(Ordering::SeqCst),
            after_first,
            "resume must not re-sample"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn b_journal_divergence_on_resume() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("div.jsonl");
        let sampler = Arc::new(MockSampler::new());
        sampler.push_text("first");
        let host: Arc<dyn SessionHost> = Arc::new(InProcessHost::new(sampler, Vec::new()));
        let script_a = r#"
            let meta = #{ name: "div", description: "d" };
            let a = agent("a");
            complete(#{ a: a });
        "#;
        let (tx, _rx) = mpsc::unbounded_channel();
        let o1 = run_workflow_on_host(
            Arc::clone(&host),
            WorkflowRunParams {
                script: script_a.into(),
                args: json!({}),
                journal: Journal::new(Some(path.clone())),
                host_tx: tx,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            },
            Some(8),
        )
        .await
        .expect("run1");
        assert!(matches!(o1, WorkflowOutcome::Completed { .. }));

        // Different prompt at same seq → divergence.
        let script_b = r#"
            let meta = #{ name: "div", description: "d" };
            let a = agent("DIFFERENT");
            complete(#{ a: a });
        "#;
        let journal = Journal::load(path).expect("load");
        let (tx2, _rx2) = mpsc::unbounded_channel();
        let o2 = run_workflow_on_host(
            host,
            WorkflowRunParams {
                script: script_b.into(),
                args: json!({}),
                journal,
                host_tx: tx2,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            },
            Some(8),
        )
        .await
        .expect("channel ok");
        match o2 {
            WorkflowOutcome::Failed { error, .. } => {
                assert!(
                    error.to_lowercase().contains("diverg") || error.contains("journal"),
                    "unexpected error: {error}"
                );
            }
            other => panic!("expected Failed on divergence, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn b_budget_reserve_exceed() {
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("a", "1");
        sampler.map_user_text("b", "2");
        let host: Arc<dyn SessionHost> = Arc::new(InProcessHost::new(sampler, Vec::new()));
        // parallel of 2 with budget 1 should fail at reserve.
        let script = r#"
            let meta = #{ name: "budget", description: "b" };
            let ws = parallel([
                #{ prompt: "a" },
                #{ prompt: "b" },
            ]);
            complete(#{ ws: ws });
        "#;
        let (tx, _rx) = mpsc::unbounded_channel();
        let outcome = run_workflow_on_host(
            host,
            WorkflowRunParams {
                script: script.into(),
                args: json!({}),
                journal: Journal::new(None),
                host_tx: tx,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            },
            Some(1),
        )
        .await
        .expect("adapter");
        assert!(
            matches!(
                outcome,
                WorkflowOutcome::BudgetExceeded { .. } | WorkflowOutcome::Failed { .. }
            ),
            "expected budget fail, got {outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn b_cancel_mid_parallel() {
        struct HoldingSampler {
            entered: tokio::sync::Notify,
        }

        #[async_trait]
        impl LlmSampler for HoldingSampler {
            async fn sample(&self, request: SampleRequest) -> Result<SampleResponse, OvoError> {
                self.entered.notify_one();
                // Park until cancelled via request token.
                request.cancel.cancelled().await;
                Err(OvoError::new(ErrorCode::LlmCancelled, "cancelled"))
            }
        }

        let holder = Arc::new(HoldingSampler {
            entered: tokio::sync::Notify::new(),
        });
        let sampler: Arc<dyn LlmSampler> = holder.clone();
        let host: Arc<dyn SessionHost> = Arc::new(InProcessHost::new(sampler, Vec::new()));
        let cancel = CancellationToken::new();
        let script = r#"
            let meta = #{ name: "cancel", description: "c" };
            let a = agent("slow");
            complete(#{ a: a });
        "#;
        let (tx, _rx) = mpsc::unbounded_channel();
        let cancel_c = cancel.clone();
        let run = tokio::spawn(async move {
            run_workflow_on_host(
                host,
                WorkflowRunParams {
                    script: script.into(),
                    args: json!({}),
                    journal: Journal::new(None),
                    host_tx: tx,
                    cancel: cancel_c,
                    max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
                },
                Some(8),
            )
            .await
        });
        holder.entered.notified().await;
        cancel.cancel();
        let outcome = run.await.expect("join").expect("adapter");
        assert!(
            matches!(
                outcome,
                WorkflowOutcome::Cancelled | WorkflowOutcome::Failed { .. }
            ),
            "expected cancel-ish outcome, got {outcome:?}"
        );
    }

    // ── A + B isomorphism ───────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ab_workflow_spend_uses_same_host_path() {
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("via-wf", "wf-out");
        // Host budget 1; workflow also budget 1 — single agent should succeed.
        let host = Arc::new(
            InProcessHost::new(sampler, Vec::new())
                .with_agent_budget(1)
                .with_max_spawn_depth(Some(8)),
        );
        let host_trait: Arc<dyn SessionHost> = host.clone();
        let script = r#"
            let meta = #{ name: "iso", description: "i" };
            let a = agent("via-wf", #{ label: "w" });
            complete(#{ a: a });
        "#;
        let (tx, _rx) = mpsc::unbounded_channel();
        let outcome = run_workflow_on_host(
            host_trait,
            WorkflowRunParams {
                script: script.into(),
                args: json!({}),
                journal: Journal::new(None),
                host_tx: tx,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            },
            Some(1),
        )
        .await
        .expect("wf");
        assert!(matches!(outcome, WorkflowOutcome::Completed { .. }));
        assert_eq!(host.agents_spent(), 1);

        // Second direct spawn should hit host budget.
        let err = host
            .spawn_agent(SpawnOpts::new("extra"))
            .await
            .expect_err("host budget");
        assert_eq!(err.code(), ErrorCode::HostBudget);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ab_fork_context_requires_messages() {
        let sampler = Arc::new(MockSampler::new());
        let host: Arc<dyn SessionHost> = Arc::new(InProcessHost::new(sampler, Vec::new()));
        let err = host
            .spawn_agent(SpawnOpts::new("x").with_fork_context(true))
            .await
            .expect_err("unsupported without messages");
        assert_eq!(err.code(), ErrorCode::HostUnsupported);
        let _ = HostError::Unsupported("fork".into());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_registry_and_fork_messages_e2e() {
        use ovo::{AgentDefinition, AgentRegistry, Instructions, Message};

        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("continue", "forked");
        let mut def = AgentDefinition::new("worker");
        def.instructions = Instructions::Static("focus".into());
        def.model = "mock".into();
        def.max_steps = 4;
        let reg = AgentRegistry::from_definitions([def]);
        let host = InProcessHost::new(sampler, Vec::new()).with_agent_registry(reg);
        let parent = vec![Message::user("history"), Message::assistant("ok")];
        let run = host
            .spawn_agent(
                SpawnOpts::new("continue")
                    .with_agent_type("worker")
                    .with_fork_messages(parent),
            )
            .await
            .expect("spawn");
        assert_eq!(run.output.as_str(), Some("forked"));
    }

    // Mode B contracts (budget conservation, await_user)

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn b_await_user_pauses_then_resume_skips() {
        use tempfile::tempdir;
        let dir = tempdir().expect("tmp");
        let path = dir.path().join("await.jsonl");
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("after-pause", "done");
        let host: Arc<dyn SessionHost> = Arc::new(InProcessHost::new(sampler, Vec::new()));
        let script = r#"
            let meta = #{ name: "await-user", description: "pause then agent" };
            await_user("user", "need human");
            let a = agent("after-pause");
            complete(#{ a: a });
        "#;
        let (tx, _rx) = mpsc::unbounded_channel();
        let path1 = path.clone();
        let outcome1 = run_workflow_on_host(
            Arc::clone(&host),
            WorkflowRunParams {
                script: script.into(),
                args: json!({}),
                journal: Journal::new(Some(path1)),
                host_tx: tx,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            },
            Some(4),
        )
        .await
        .expect("run1");
        assert!(
            matches!(outcome1, WorkflowOutcome::Paused { .. }),
            "{outcome1:?}"
        );

        let journal = Journal::load(path.clone()).expect("load");
        assert_eq!(journal.len(), 1, "only await_user journaled");
        let (tx2, _rx2) = mpsc::unbounded_channel();
        let outcome2 = run_workflow_on_host(
            host,
            WorkflowRunParams {
                script: script.into(),
                args: json!({}),
                journal,
                host_tx: tx2,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            },
            Some(4),
        )
        .await
        .expect("run2");
        assert!(
            matches!(outcome2, WorkflowOutcome::Completed { .. }),
            "{outcome2:?}"
        );
        let journal2 = Journal::load(path).expect("load2");
        assert_eq!(journal2.len(), 2, "await_user + agent");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn b_budget_exceeded_on_host_does_not_double_charge_on_resume() {
        use tempfile::tempdir;
        // Host budget 0 → first agent fails BudgetExceeded; journal must stay empty so resume can retry with higher budget.
        let dir = tempdir().expect("tmp");
        let path = dir.path().join("budget.jsonl");
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("task", "ok");
        let sampler_dyn: Arc<dyn LlmSampler> = sampler;
        let host0: Arc<dyn SessionHost> =
            Arc::new(InProcessHost::new(Arc::clone(&sampler_dyn), Vec::new()).with_agent_budget(0));
        let script = r#"
            let meta = #{ name: "budget-resume", description: "b" };
            let a = agent("task");
            complete(#{ a: a });
        "#;
        let (tx, _rx) = mpsc::unbounded_channel();
        let path1 = path.clone();
        let outcome1 = run_workflow_on_host(
            host0,
            WorkflowRunParams {
                script: script.into(),
                args: json!({}),
                journal: Journal::new(Some(path1)),
                host_tx: tx,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            },
            Some(8),
        )
        .await
        .expect("run1");
        assert!(
            matches!(outcome1, WorkflowOutcome::BudgetExceeded { .. }),
            "{outcome1:?}"
        );
        let journal = Journal::load(path.clone()).expect("load");
        assert_eq!(
            journal.len(),
            0,
            "budget failure must not journal interrupted agent"
        );

        let host1: Arc<dyn SessionHost> =
            Arc::new(InProcessHost::new(sampler_dyn, Vec::new()).with_agent_budget(2));
        let (tx2, _rx2) = mpsc::unbounded_channel();
        let outcome2 = run_workflow_on_host(
            host1,
            WorkflowRunParams {
                script: script.into(),
                args: json!({}),
                journal: Journal::load(path.clone()).expect("load2"),
                host_tx: tx2,
                cancel: CancellationToken::new(),
                max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
            },
            Some(8),
        )
        .await
        .expect("run2");
        assert!(
            matches!(outcome2, WorkflowOutcome::Completed { .. }),
            "{outcome2:?}"
        );
        assert_eq!(Journal::load(path).expect("final").len(), 1);
    }

    // agent resolution / host

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn w5_builtin_types_spawnable() {
        use ovo::{EXPLORE, GENERAL_PURPOSE, PLAN};

        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("gp", "ok-gp");
        sampler.map_user_text("ex", "ok-ex");
        sampler.map_user_text("pl", "ok-pl");
        let host = InProcessHost::new(sampler, Vec::new());
        for (prompt, ty, expect) in [
            ("gp", GENERAL_PURPOSE, "ok-gp"),
            ("ex", EXPLORE, "ok-ex"),
            ("pl", PLAN, "ok-pl"),
        ] {
            let run = host
                .spawn_agent(SpawnOpts::new(prompt).with_agent_type(ty))
                .await
                .unwrap_or_else(|e| panic!("spawn {ty}: {e}"));
            assert!(run.success, "{ty}");
            assert_eq!(run.output.as_str(), Some(expect), "{ty}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn w5_fork_from_parent_handle_and_resume_store() {
        use std::path::PathBuf;

        use ovo::{
            ChatStateHandle, MemoryWorkflowRunStore, Message, WorkflowRunRecord, WorkflowRunStore,
        };

        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("continue", "from-handle");
        let parent =
            ChatStateHandle::spawn(vec![Message::user("history"), Message::assistant("prior")]);
        let store = Arc::new(MemoryWorkflowRunStore::new());
        let mut rec = WorkflowRunRecord::new_running("run-1", "done-wf", PathBuf::from("j.jsonl"));
        rec.apply_outcome(&WorkflowOutcome::Completed {
            result: serde_json::json!({"ok": true}),
        });
        store.put(rec).expect("put");

        let host = InProcessHost::new(sampler, Vec::new())
            .with_parent_handle(parent)
            .with_run_store(store);

        let forked = host
            .spawn_agent(SpawnOpts::new("continue").with_fork_context(true))
            .await
            .expect("fork handle");
        assert_eq!(forked.output.as_str(), Some("from-handle"));

        let resumed = host
            .spawn_agent(SpawnOpts::new("ignored").with_resume_from("run-1"))
            .await
            .expect("resume");
        assert!(resumed.success);
        assert_eq!(resumed.label.as_deref(), Some("done-wf"));
        assert_eq!(
            resumed.output,
            serde_json::json!({"ok": true}),
            "resume_from must return stored Completed.result"
        );
        assert_eq!(host.agents_spent(), 2, "fork + resume each charge budget");
    }

    #[cfg(feature = "toolkit")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sandboxed_host_rejects_missing_jail() {
        use std::path::PathBuf;

        use ovo::{NoSandbox, sandboxed_host};

        let sampler: Arc<dyn LlmSampler> = Arc::new(MockSampler::new());
        let err = sandboxed_host(
            sampler,
            Arc::new(NoSandbox),
            PathBuf::from("/no/such/ovo-sandboxed-host-jail"),
        )
        .expect_err("missing jail");
        assert_eq!(err.code(), ErrorCode::HostIsolation);
    }

    // state / ledger

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn w4_ledger_and_jsonl_restart() {
        use tempfile::tempdir;

        let dir = tempdir().expect("tmp");
        let store = JsonlPersistence::new(dir.path().join("sess")).with_snapshot_every(2);
        let h = ChatStateHandle::spawn(vec![Message::system("s")]);
        h.append(Message::user("p0")).await.expect("p0");
        h.record_main_usage_model(Usage::new(10, 5), "mock-a").await;
        h.append(Message::assistant("a0")).await.expect("a0");
        h.append(Message::user("p1")).await.expect("p1");
        h.record_main_usage_model(Usage::new(3, 2), "mock-b").await;
        h.record_subagent_usage(Usage::new(1, 1)).await;
        h.record_compaction_at("max_messages").await;

        let snap = h.snapshot().await.expect("snap");
        assert_eq!(snap.prompt_index, vec![1, 3]);
        assert_eq!(snap.usage.main.total_tokens, 20);
        assert_eq!(snap.usage.subagents.total_tokens, 2);
        assert_eq!(snap.usage.per_prompt.len(), 2);
        assert_eq!(
            snap.usage.per_prompt.first().map(|u| u.total_tokens),
            Some(15)
        );
        assert!(snap.usage.per_model.contains_key("mock-a"));
        assert!(snap.usage.per_model.contains_key("mock-b"));
        assert_eq!(snap.usage.compaction_at.len(), 1);

        h.save_to(&store).await.expect("save");
        h.shutdown().await;

        // Restart: replay JSONL events + ledger from snapshot.
        let h2 = ChatStateHandle::open_or_new(&store).await.expect("open");
        let loaded = h2.snapshot().await.expect("loaded");
        assert_eq!(loaded.messages.len(), 4);
        assert_eq!(loaded.prompt_index, vec![1, 3]);
        assert_eq!(loaded.usage.main.total_tokens, 20);
        assert_eq!(loaded.usage.compaction_at.len(), 1);
        h2.shutdown().await;
    }
}
