//! Ovo — embeddable multi-agent runtime kernel.
//!
//! | Crate | Role |
//! |-------|------|
//! | `ovo-types` | ids, messages, usage, errors |
//! | `ovo-protocol` | tool id, content blocks, span catalogue |
//! | `ovo-obs` | metrics sink, redact, recording / prometheus text |
//! | `ovo-tools` | `DynTool`, stream, dispatch, approval |
//! | `ovo-toolkit` | cwd-jailed fs/shell tools (feature) |
//! | `ovo-llm` | sampler + mock / openai / ollama |
//! | `ovo-agent` | definition, builder, discovery |
//! | `ovo-state` | conversation handle, ledger, persistence |
//! | `ovo-compaction` | compaction strategies |
//! | `ovo-runtime` | turn, session, host, workflow adapter |
//! | `ovo-workflow` | Rhai engine, journal, validate (no LLM) |
//!
//! Session / handle → `TurnRuntime` → tools → gates → metrics →
//! `SessionHost` spawn and/or journaled workflow.
//!
//! Optional host capabilities (e.g. `git_diff_since`) require explicit setup.

#![forbid(unsafe_code)]
// Feature-gated transitive deps are unused when compiling the facade lib alone.
#![allow(
    unused_crate_dependencies,
    reason = "facade re-exports optional workspace crates"
)]

#[cfg(feature = "runtime")]
pub use ovo_agent as agent;
#[cfg(feature = "runtime")]
pub use ovo_agent::{
    Agent, AgentBuilder, AgentDefinition, AgentRegistry, AgentSource, CompletionRequirement,
    EXPLORE, GENERAL_PURPOSE, IdentityAssembler, Instructions, ORCHESTRATOR_DELEGATION_PROMPT,
    PLAN, PROJECT_AGENTS_DIR, PROJECT_AGENTS_MD, ProjectPromptAssembler, PromptAssembler,
    ToolPolicy, USER_AGENTS_DIR, agents_md_path, builtin_definitions, builtin_names,
    by_name_in_dir, by_name_resolved, discover_in_dir, discover_project, discover_user, load_file,
    parse_definition_markdown, project_agent_dirs, resolve_agents, user_agents_dir,
};
#[cfg(feature = "compaction")]
pub use ovo_compaction as compaction;
#[cfg(feature = "runtime")]
pub use ovo_llm as llm;
#[cfg(feature = "openai")]
pub use ovo_llm::OpenAiCompatSampler;
#[cfg(feature = "runtime")]
pub use ovo_llm::{
    Admission, BreakerConfig, BreakerOutcome, BreakerSampler, BreakerState, CircuitBreaker,
    DEFAULT_IDLE_TIMEOUT, DEFAULT_MAX_ATTEMPTS, HttpRetryClass, LlmSampler, MAX_RETRY_AFTER,
    MAX_RETRY_BACKOFF, MockSampler, OpenAiCompatConfig, RATE_LIMIT_RETRY_THRESHOLD, RetryContext,
    RetryDecision, RetryPolicy, RetryingSampler, SampleEvent, SampleRequest, SampleResponse,
    SampleStream, ToolChoice, backoff_for_attempt, build_chat_completions_body,
    classify_http_status, decide_retry, error_code_for_http, is_empty_response,
    parse_chat_completions_response, response_to_stream,
};
#[cfg(feature = "ollama")]
pub use ovo_llm::{OllamaConfig, OllamaSampler};
#[cfg(feature = "obs")]
pub use ovo_obs as obs;
#[cfg(feature = "obs")]
pub use ovo_obs::{
    PrometheusRecorder, REDACTED, RecordingMetrics, emit_catalogue_smoke, looks_like_secret_key,
    metric_catalogue_snapshot, redact_key_value, redact_map, required_metric_names,
    required_span_names,
};
pub use ovo_protocol as protocol;
pub use ovo_protocol::{
    ContentBlock, IMAGE_TOKEN_COST, ImageBlock, MESSAGE_FRAME_TOKENS, PreflightOverflow,
    SPAN_COMPACT, SPAN_SAMPLE, SPAN_SESSION, SPAN_SPAWN, SPAN_TOOL, SPAN_TOOL_BATCH, SPAN_TURN,
    SPAN_WORKFLOW, SPAN_WORKFLOW_HOST, ToolId, TurnEvent, TurnEventKind, check_context_overflow,
    estimate_image_tokens, estimate_text_tokens, span_catalogue_snapshot,
};
#[cfg(feature = "runtime")]
pub use ovo_runtime as runtime;
#[cfg(feature = "runtime")]
pub use ovo_runtime::{
    AgentRunResult, CompactionOutcome, CompactionStrategy, CompletionToolGate, ConversationState,
    DEFAULT_MAX_CONCURRENT_CHILDREN, DEFAULT_MAX_SPAWN_DEPTH, EventBus, EventSink, GateChain,
    GateDecision, HARD_STOP_THRESHOLD, InProcessHost, InProcessIsolation, IsolationBackend,
    IsolationEnv, LifecycleFanout, MaxMessages, MetricsSink, NUDGE_THRESHOLD, NoopLifecycle,
    NoopMetrics, Session, SessionHost, SharedMetrics, SpawnAgentTool, SpawnOpts,
    StationarityAction, StationarityTracker, StopGate, TokenThreshold, TurnAbortReason, TurnInput,
    TurnLifecycleContributor, TurnOptions, TurnOutcome, TurnRuntime, VecConversationState,
    estimate_conversation_tokens, evaluate_stop_gates, fingerprint_batch, isolation_error,
    nudge_message,
};
#[cfg(all(feature = "runtime", feature = "workflow"))]
pub use ovo_runtime::{
    WorkflowSideEffects, run_workflow_configured, run_workflow_configured_with_events,
    run_workflow_on_host, run_workflow_on_host_with_metrics,
};
#[cfg(feature = "sandbox")]
pub use ovo_sandbox as sandbox;
#[cfg(all(feature = "landlock", target_os = "linux"))]
pub use ovo_sandbox::LandlockBackend;
#[cfg(feature = "sandbox")]
pub use ovo_sandbox::{
    AllowAllExecPolicy, DenyAllExecPolicy, ExecDecision, ExecPolicy, FsPolicy, NetPolicy,
    NoSandbox, PrefixExecPolicy, PrefixRule, SandboxBackend, SandboxError, SandboxPolicy,
    SharedExecPolicy, TrustedExecution,
};
#[cfg(all(feature = "seatbelt", target_os = "macos"))]
pub use ovo_sandbox::{SANDBOX_EXEC, SeatbeltBackend};
#[cfg(feature = "state")]
pub use ovo_state as state;
#[cfg(feature = "state")]
pub use ovo_state::{
    ChatPersistence, ChatStateHandle, ChatStateSnapshot, CompactionRecord, DEFAULT_SESSIONS_DIR,
    DEFAULT_SNAPSHOT_EVERY, EVENTS_HEADER, FilePersistence, InMemoryMemory, JsonlPersistence,
    MemoryItem, MemoryPersistence, MemoryPort, NullMemory, NullPersistence, UsageLedger,
    check_tool_pairing, default_session_path, messages_only, session_jsonl_dir,
};
#[cfg(feature = "toolkit")]
pub use ovo_toolkit as toolkit;
#[cfg(feature = "toolkit")]
pub use ovo_toolkit::{
    GlobTool, GrepTool, ReadFileTool, ShellTool, WriteFileTool, default_toolkit, glob_match,
    resolve_jailed,
};
#[cfg(feature = "runtime")]
pub use ovo_tools as tools;
#[cfg(feature = "runtime")]
pub use ovo_tools::{
    AlwaysDeny, ApprovalDecision, ApprovalGate, ApprovalPolicy, AutoApprove, CalcTool,
    CapabilityFlag, CapabilityMode, ConcurrencyMode, Destructiveness, DispatchOutcome,
    DispatchRequest, DynTool, EXTRA_SPAWN_DEPTH, InterruptBehavior, MAX_DELTA_BYTES,
    MAX_FRAME_BYTES, SharedTool, StaticToolSource, ToolCallContext, ToolDefinition, ToolDispatch,
    ToolError, ToolMetadata, ToolProgress, ToolRegistry, ToolResult, ToolSource, ToolStream,
    ToolStreamItem, drain_terminal, drain_with_progress, merge_arc_sources, merge_tool_sources,
    partial_progress_frames, terminal_only, with_progress,
};
pub use ovo_types as types;
pub use ovo_types::{
    AgentId, CompletionTokensDetails, ContentPart, Deadline, ErrorCode, ImageMime, Message,
    OvoError, PromptTokensDetails, Result, RetryClass, Role, RunId, SessionId, ToolCall,
    ToolCallId, Usage, WorkflowRunId,
};
#[cfg(feature = "workflow")]
pub use ovo_workflow as workflow;
#[cfg(feature = "workflow")]
pub use ovo_workflow::{
    AgentOpts, AgentResult as WorkflowAgentResult, BudgetState, DEFAULT_AGENT_BUDGET,
    FileWorkflowRunStore, HostError, Journal, JournalEntry, JournalError, MAX_AGENT_BUDGET,
    MAX_JOURNAL_BYTES, MAX_JOURNAL_ENTRIES, MemoryWorkflowRunStore, PauseKind, StoreError,
    ValidationError, ValidationReport, WorkflowHostRequest, WorkflowMeta, WorkflowOutcome,
    WorkflowRunParams, WorkflowRunRecord, WorkflowRunStatus, WorkflowRunStore, default_probe_args,
    extract_meta, request_hash, run_workflow, validate_script, validate_script_with_agent_budget,
};
