//! Turn runtime and session host for the Ovo kernel.

#![forbid(unsafe_code)]
// Dev-dependencies (toolkit, tempfile) are only used by integration tests.
#![allow(
    unused_crate_dependencies,
    reason = "dev-deps for integration tests appear unused on lib target"
)]

pub mod events;
pub mod gates;
pub mod host;
pub mod isolation;
pub mod lifecycle;
pub mod metrics;
pub mod schema;
pub mod session;
pub mod side_effects;
pub mod spawn_tool;
pub mod state;
pub mod stationarity;
pub mod turn;
pub mod workflow_host;

pub use events::EventSink;
pub use gates::{CompletionToolGate, GateChain, GateDecision, StopGate, evaluate_stop_gates};
pub use host::{
    AgentRunResult, ChildToolkit, DEFAULT_MAX_CONCURRENT_CHILDREN, DEFAULT_MAX_SPAWN_DEPTH,
    InProcessHost, SessionHost, SpawnOpts,
};
pub use isolation::{InProcessIsolation, IsolationBackend, IsolationEnv, isolation_error};
pub use lifecycle::{LifecycleFanout, NoopLifecycle, TurnAbortReason, TurnLifecycleContributor};
pub use metrics::{
    METRIC_COMPACTIONS_TOTAL, METRIC_SAMPLE_DURATION_MS, METRIC_SPAWNS_TOTAL, METRIC_TOKENS_TOTAL,
    METRIC_TOOL_CALLS_TOTAL, METRIC_TOOL_DURATION_MS, METRIC_TURN_DURATION_MS, METRIC_TURN_STEPS,
    METRIC_TURNS_TOTAL, METRIC_WORKFLOW_AGENTS_TOTAL, METRIC_WORKFLOW_RUNS_TOTAL, MetricsSink,
    NoopMetrics, SharedMetrics, record_compaction, record_sample, record_spawn, record_tool_call,
    record_turn, record_workflow_agents, record_workflow_run, required_metric_names,
};
pub use ovo_compaction::{CompactionOutcome, CompactionStrategy, MaxMessages, TokenThreshold};
pub use ovo_protocol::{TurnEvent, TurnEventKind};
pub use ovo_sandbox::TrustedExecution;
pub use ovo_tools::EventBus;
pub use schema::{
    STRUCTURED_OUTPUT_MAX_RETRIES, compile_schema, schema_retry_reminder,
    validate_structured_output,
};
pub use session::Session;
pub use side_effects::WorkflowSideEffects;
pub use spawn_tool::SpawnAgentTool;
pub use state::{ConversationState, VecConversationState, estimate_messages_tokens};
pub use stationarity::{
    HARD_STOP_THRESHOLD, NUDGE_THRESHOLD, StationarityAction, StationarityTracker,
    fingerprint_batch, nudge_message,
};
pub use turn::{TurnInput, TurnOptions, TurnOutcome, TurnRuntime, estimate_conversation_tokens};
pub use workflow_host::{
    run_workflow_configured, run_workflow_configured_with_events, run_workflow_on_host,
    run_workflow_on_host_with_metrics,
};
