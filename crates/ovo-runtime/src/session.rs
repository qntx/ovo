//! Multi-turn session helper over [`TurnRuntime`].
//!
//! # Conversation state model (canonical)
//!
//! - **Turn-local buffer:** [`VecConversationState`] implements
//!   [`ConversationState`] for the synchronous sample/tool loop.
//! - **Session source of truth (optional):** [`ChatStateHandle`] is the
//!   multi-turn durable surface. Prefer
//!   [`Session::run_turn_on_handle`] when you need actor isolation,
//!   usage ledger, or checkpointing via [`ovo_state::ChatPersistence`].
//!
//! Do not treat the two as parallel “equal” product APIs: handle is the
//! session store; `VecConversationState` is the turn engine buffer.

use std::time::Instant;

use ovo_agent::Agent;
use ovo_llm::LlmSampler;
use ovo_state::{ChatPersistence, ChatStateHandle};
use ovo_types::OvoError;

use crate::metrics::{MetricsSink, NoopMetrics, record_turn};
use crate::state::{ConversationState, VecConversationState};
use crate::turn::{TurnInput, TurnOptions, TurnOutcome, TurnRuntime};

/// Thin multi-turn orchestrator (one agent, persistent conversation state).
#[derive(Debug, Clone, Copy)]
pub struct Session {
    runtime: TurnRuntime,
    turn_count: u64,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    /// Create a session.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            runtime: TurnRuntime::new(),
            turn_count: 0,
        }
    }

    /// Number of turns completed successfully (or cancelled) through this session.
    #[must_use]
    pub const fn turn_count(self) -> u64 {
        self.turn_count
    }

    /// Run one user turn, appending to `state`.
    ///
    /// Does not rewrite [`TurnOptions`]. Production embedders must pass
    /// [`TurnOptions::for_host`] themselves; [`TurnOptions::default`] stays
    /// `AutoApprove` for unit tests.
    ///
    /// # Errors
    ///
    /// Propagates [`TurnRuntime::run`] failures.
    pub async fn run_turn(
        &mut self,
        agent: &Agent,
        sampler: &dyn LlmSampler,
        state: &mut dyn ConversationState,
        input: TurnInput,
        options: TurnOptions,
    ) -> Result<TurnOutcome, OvoError> {
        self.run_turn_with_metrics(agent, sampler, state, input, options, &NoopMetrics)
            .await
    }

    /// Like [`Self::run_turn`] with metrics.
    ///
    /// # Errors
    ///
    /// Propagates turn failures.
    pub async fn run_turn_with_metrics(
        &mut self,
        agent: &Agent,
        sampler: &dyn LlmSampler,
        state: &mut dyn ConversationState,
        input: TurnInput,
        options: TurnOptions,
        metrics: &dyn MetricsSink,
    ) -> Result<TurnOutcome, OvoError> {
        let started = Instant::now();
        let outcome = self
            .runtime
            .run(agent, sampler, state, input, options)
            .await;
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        match &outcome {
            Ok(o) => {
                self.turn_count = self.turn_count.saturating_add(1);
                let status = if o.cancelled { "cancelled" } else { "ok" };
                let steps = u64::try_from(o.steps).unwrap_or(u64::MAX);
                record_turn(metrics, status, steps, ms);
            }
            Err(_) => {
                record_turn(metrics, "error", 0, ms);
            }
        }
        outcome
    }

    /// Run a turn against a [`ChatStateHandle`]: snapshot → local turn → replace + usage.
    ///
    /// The turn loop remains synchronous over a local buffer; the actor is the
    /// durable source of truth between turns.
    ///
    /// # Errors
    ///
    /// Propagates turn or handle channel failures.
    pub async fn run_turn_on_handle(
        &mut self,
        agent: &Agent,
        sampler: &dyn LlmSampler,
        handle: &ChatStateHandle,
        input: TurnInput,
        options: TurnOptions,
    ) -> Result<TurnOutcome, OvoError> {
        self.run_turn_on_handle_with_metrics(agent, sampler, handle, input, options, &NoopMetrics)
            .await
    }

    /// [`Self::run_turn_on_handle`] with metrics.
    ///
    /// # Errors
    ///
    /// Propagates turn failures.
    pub async fn run_turn_on_handle_with_metrics(
        &mut self,
        agent: &Agent,
        sampler: &dyn LlmSampler,
        handle: &ChatStateHandle,
        input: TurnInput,
        options: TurnOptions,
        metrics: &dyn MetricsSink,
    ) -> Result<TurnOutcome, OvoError> {
        self.run_turn_on_handle_checkpointed(agent, sampler, handle, input, options, metrics, None)
            .await
    }

    /// Session turn with optional durable checkpoint after each turn.
    ///
    /// When `store` is `Some`, the handle snapshot is written after the turn
    /// (including failed turns that still mutated history).
    ///
    /// # Errors
    ///
    /// Turn failures or persistence I/O.
    #[allow(
        clippy::too_many_arguments,
        reason = "session turn needs agent, sampler, handle, I/O, metrics, optional store"
    )]
    pub async fn run_turn_on_handle_checkpointed(
        &mut self,
        agent: &Agent,
        sampler: &dyn LlmSampler,
        handle: &ChatStateHandle,
        input: TurnInput,
        options: TurnOptions,
        metrics: &dyn MetricsSink,
        store: Option<&dyn ChatPersistence>,
    ) -> Result<TurnOutcome, OvoError> {
        let snap = handle.snapshot().await?;
        let mut local = VecConversationState::from_messages(snap.messages);
        let outcome = self
            .run_turn_with_metrics(agent, sampler, &mut local, input, options, metrics)
            .await;
        // Always fold local history back into the handle (session SoT).
        handle.replace(local.messages().to_vec()).await;
        if let Ok(o) = &outcome {
            handle.record_main_usage(o.usage).await;
            if o.cancelled {
                handle.mark_incomplete().await;
            }
        } else {
            handle.mark_incomplete().await;
        }
        if let Some(store) = store {
            handle.save_to(store).await?;
        }
        outcome
    }

    /// Open a handle from `store` (or empty), run one checkpointed turn, return handle + outcome.
    ///
    /// Convenience for hosts that own a single file/session path.
    ///
    /// # Errors
    ///
    /// Load / turn / save failures.
    #[allow(
        clippy::too_many_arguments,
        reason = "open+turn packs agent, sampler, store, input, options, metrics"
    )]
    pub async fn open_checkpointed_turn(
        &mut self,
        agent: &Agent,
        sampler: &dyn LlmSampler,
        store: &dyn ChatPersistence,
        input: TurnInput,
        options: TurnOptions,
        metrics: &dyn MetricsSink,
    ) -> Result<(ChatStateHandle, TurnOutcome), OvoError> {
        let handle = ChatStateHandle::open_or_new(store).await?;
        let outcome = self
            .run_turn_on_handle_checkpointed(
                agent,
                sampler,
                &handle,
                input,
                options,
                metrics,
                Some(store),
            )
            .await?;
        Ok((handle, outcome))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ovo_agent::AgentBuilder;
    use ovo_llm::MockSampler;
    use ovo_state::ChatStateHandle;

    use super::*;
    use crate::state::VecConversationState;

    #[tokio::test]
    async fn multi_turn() {
        let sampler = Arc::new(MockSampler::new());
        sampler.push_text("one");
        sampler.push_text("two");
        let agent = AgentBuilder::named("a")
            .model("mock")
            .build()
            .expect("agent");
        let mut state = VecConversationState::new();
        let mut session = Session::new();
        let o1 = session
            .run_turn(
                &agent,
                sampler.as_ref(),
                &mut state,
                TurnInput::Text("hi".into()),
                TurnOptions::default(),
            )
            .await
            .expect("t1");
        let o2 = session
            .run_turn(
                &agent,
                sampler.as_ref(),
                &mut state,
                TurnInput::Text("again".into()),
                TurnOptions::default(),
            )
            .await
            .expect("t2");
        assert_eq!(o1.output_text, "one");
        assert_eq!(o2.output_text, "two");
        assert_eq!(session.turn_count(), 2);
        assert!(state.messages().len() >= 4);
    }

    #[tokio::test]
    async fn multi_turn_on_handle() {
        let sampler = Arc::new(MockSampler::new());
        sampler.push_text("alpha");
        sampler.push_text("beta");
        let agent = AgentBuilder::named("a")
            .model("mock")
            .build()
            .expect("agent");
        let handle = ChatStateHandle::spawn(vec![]);
        let mut session = Session::new();
        let o1 = session
            .run_turn_on_handle(
                &agent,
                sampler.as_ref(),
                &handle,
                TurnInput::Text("hi".into()),
                TurnOptions::default(),
            )
            .await
            .expect("t1");
        let o2 = session
            .run_turn_on_handle(
                &agent,
                sampler.as_ref(),
                &handle,
                TurnInput::Text("again".into()),
                TurnOptions::default(),
            )
            .await
            .expect("t2");
        assert_eq!(o1.output_text, "alpha");
        assert_eq!(o2.output_text, "beta");
        let snap = handle.snapshot().await.expect("snapshot");
        assert!(snap.messages.len() >= 4);
        assert!(snap.usage.main.total_tokens > 0);
        handle.shutdown().await;
    }

    #[tokio::test]
    async fn handle_checkpoint_to_memory() {
        use ovo_state::MemoryPersistence;

        let sampler = Arc::new(MockSampler::new());
        sampler.push_text("ckpt");
        let agent = AgentBuilder::named("a")
            .model("mock")
            .build()
            .expect("agent");
        let handle = ChatStateHandle::spawn(vec![]);
        let store = MemoryPersistence::default();
        let mut session = Session::new();
        session
            .run_turn_on_handle_checkpointed(
                &agent,
                sampler.as_ref(),
                &handle,
                TurnInput::Text("hi".into()),
                TurnOptions::default(),
                &NoopMetrics,
                Some(&store),
            )
            .await
            .expect("turn");
        let loaded = store.load().await.expect("load").expect("some");
        assert!(!loaded.messages.is_empty());
        handle.shutdown().await;
    }

    #[tokio::test]
    async fn open_checkpointed_turn_round_trip() {
        use ovo_state::MemoryPersistence;

        let sampler = Arc::new(MockSampler::new());
        sampler.push_text("one");
        sampler.push_text("two");
        let agent = AgentBuilder::named("a")
            .model("mock")
            .build()
            .expect("agent");
        let store = MemoryPersistence::default();
        let mut session = Session::new();
        let (h1, o1) = session
            .open_checkpointed_turn(
                &agent,
                sampler.as_ref(),
                &store,
                TurnInput::Text("a".into()),
                TurnOptions::default(),
                &NoopMetrics,
            )
            .await
            .expect("t1");
        assert_eq!(o1.output_text, "one");
        h1.shutdown().await;

        let (h2, o2) = session
            .open_checkpointed_turn(
                &agent,
                sampler.as_ref(),
                &store,
                TurnInput::Text("b".into()),
                TurnOptions::default(),
                &NoopMetrics,
            )
            .await
            .expect("t2");
        assert_eq!(o2.output_text, "two");
        // History includes both turns after reload.
        assert!(h2.messages().await.expect("msgs").len() >= 4);
        h2.shutdown().await;
    }
}
