//! Model-facing `spawn_agent` tool for dynamic multi-agent delegation.

use std::sync::Arc;

use async_trait::async_trait;
use ovo_tools::registry::CapabilityMode;
use ovo_tools::{DynTool, ToolCallContext, ToolMetadata, ToolResult};
use ovo_types::{ErrorCode, OvoError};
use serde_json::{Value, json};

use crate::host::{SessionHost, SpawnOpts};

/// Tool that spawns a nested agent through a shared [`SessionHost`].
///
/// This is the **dynamic delegation** entry point for a parent agent’s `ReAct`
/// loop: the model calls `spawn_agent`, the tool blocks until the child finishes,
/// and the child’s output is returned as the tool result.
///
/// Attach this tool to the **parent** agent only (or share the host via
/// [`std::sync::Arc::new_cyclic`]). Do not construct a second
/// [`crate::InProcessHost`] for children; children reuse this host (and
/// optional [`crate::ChildToolkit`]).
///
/// Nesting depth is derived from [`ToolCallContext::spawn_depth`]: a top-level
/// session turn spawns at depth `0`; a host-spawned agent at depth `d` spawns at
/// `d + 1`.
pub struct SpawnAgentTool {
    host: Arc<dyn SessionHost>,
    default_capability: CapabilityMode,
    /// When set, only these `agent_type` values may be spawned (fail-closed).
    allowed_agent_types: Option<Vec<String>>,
}

impl std::fmt::Debug for SpawnAgentTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpawnAgentTool")
            .field("default_capability", &self.default_capability)
            .field("allowed_agent_types", &self.allowed_agent_types)
            .finish_non_exhaustive()
    }
}

impl SpawnAgentTool {
    /// Bind to a host.
    #[must_use]
    pub fn new(host: Arc<dyn SessionHost>) -> Self {
        Self {
            host,
            default_capability: CapabilityMode::Full,
            allowed_agent_types: None,
        }
    }

    /// Default capability mode for children when the model omits it.
    #[must_use]
    pub const fn with_default_capability(mut self, mode: CapabilityMode) -> Self {
        self.default_capability = mode;
        self
    }

    /// Restrict spawnable `agent_type` values. Empty allowlist rejects all typed spawns.
    #[must_use]
    pub fn with_allowed_agent_types(
        mut self,
        types: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.allowed_agent_types = Some(types.into_iter().map(Into::into).collect());
        self
    }
}

#[async_trait]
impl DynTool for SpawnAgentTool {
    fn name(&self) -> &'static str {
        "spawn_agent"
    }

    fn description(&self) -> &'static str {
        "Spawn a nested agent to handle a subtask with its own context. \
         Provide a clear prompt; optionally set label, capability_mode \
         (full | read_only | plan), agent_type, max_steps, and output_schema."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Task instructions for the nested agent"
                },
                "label": {
                    "type": "string",
                    "description": "Optional short label for logs and aggregation"
                },
                "capability_mode": {
                    "type": "string",
                    "enum": ["full", "read_only", "plan"],
                    "description": "Tool capability filter for the child agent"
                },
                "max_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional max ReAct steps for the child turn"
                },
                "agent_type": {
                    "type": "string",
                    "description": "Optional registered agent definition name"
                },
                "output_schema": {
                    "type": "object",
                    "description": "Optional JSON schema for structured child output"
                },
                "max_output_tokens": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional max output tokens for the child sampler"
                }
            },
            "required": ["prompt"],
            "additionalProperties": false
        })
    }

    fn metadata(&self) -> ToolMetadata {
        ToolMetadata::spawn()
    }

    async fn call(&self, ctx: ToolCallContext, arguments: Value) -> Result<ToolResult, OvoError> {
        if ctx.is_cancelled() {
            return Err(OvoError::new(
                ErrorCode::ToolCancelled,
                "spawn_agent cancelled",
            ));
        }

        let prompt = arguments
            .get("prompt")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                OvoError::new(ErrorCode::ToolInvalidArgs, "spawn_agent requires prompt")
            })?;

        let label = arguments
            .get("label")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let capability_mode = match arguments.get("capability_mode").and_then(Value::as_str) {
            Some(mode) => parse_capability(mode)?,
            None => self.default_capability,
        };
        let max_steps = arguments
            .get("max_steps")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok());
        let agent_type = arguments
            .get("agent_type")
            .and_then(Value::as_str)
            .map(str::to_owned);
        if let Some(allow) = &self.allowed_agent_types {
            let Some(ref t) = agent_type else {
                return Err(OvoError::new(
                    ErrorCode::ToolInvalidArgs,
                    "spawn_agent requires agent_type when an allowlist is configured",
                ));
            };
            if !allow.iter().any(|a| a == t) {
                return Err(OvoError::new(
                    ErrorCode::ToolInvalidArgs,
                    format!("agent_type '{t}' is not in the spawn allowlist"),
                ));
            }
        }
        let output_schema = arguments.get("output_schema").cloned();
        let max_output_tokens = arguments.get("max_output_tokens").and_then(Value::as_u64);

        // Always link to the parent token (never invent a root token — TOCTOU-safe).
        let child_cancel = ctx.cancel.child_token();

        // Top-level session → depth 0; host-spawned agent at d → child at d+1.
        let depth = ctx.spawn_depth().map_or(0, |d| d.saturating_add(1));

        let mut opts = SpawnOpts::new(prompt)
            .with_capability(capability_mode)
            .with_cancel(child_cancel)
            .with_depth(depth);
        if let Some(bus) = ctx.events.clone() {
            opts = opts.with_events(bus);
        }
        if let Some(label) = label {
            opts = opts.with_label(label);
        }
        if let Some(max_steps) = max_steps {
            opts = opts.with_max_steps(max_steps);
        }
        if let Some(agent_type) = agent_type {
            opts = opts.with_agent_type(agent_type);
        }
        if let Some(schema) = output_schema {
            opts = opts.with_output_schema(schema);
        }
        if let Some(n) = max_output_tokens {
            opts = opts.with_max_output_tokens(n);
        }

        let run = self.host.spawn_agent(opts).await.map_err(|e| {
            OvoError::new(ErrorCode::ToolExecution, e.message().to_owned()).with_source(e)
        })?;

        let content = serde_json::to_string_pretty(&json!({
            "agent_id": run.agent_id.to_string(),
            "label": run.label,
            "success": run.success,
            "cancelled": run.cancelled,
            "output": run.output,
            "steps": run.steps,
            "duration_ms": run.duration_ms,
        }))
        .unwrap_or_else(|_| run.output.to_string());

        Ok(ToolResult {
            content,
            structured: Some(json!({
                "agent_id": run.agent_id.to_string(),
                "label": run.label,
                "success": run.success,
                "output": run.output,
            })),
            is_error: !run.success || run.cancelled,
        })
    }
}

fn parse_capability(mode: &str) -> Result<CapabilityMode, OvoError> {
    CapabilityMode::parse(mode).ok_or_else(|| {
        OvoError::new(
            ErrorCode::ToolInvalidArgs,
            format!("unknown capability_mode '{mode}' (expected full|read_only|plan)"),
        )
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ovo_llm::MockSampler;
    use serde_json::json;

    use super::*;
    use crate::host::InProcessHost;

    #[tokio::test]
    async fn spawn_tool_runs_child() {
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
        let structured = result.structured.expect("structured");
        assert_eq!(
            structured.get("output").and_then(|v| v.as_str()),
            Some("child-done")
        );
        assert_eq!(structured.get("label").and_then(|v| v.as_str()), Some("w1"));
    }

    #[tokio::test]
    async fn spawn_tool_depth_fail_closed() {
        let sampler = Arc::new(MockSampler::new());
        let host: Arc<dyn SessionHost> =
            Arc::new(InProcessHost::new(sampler, Vec::new()).with_max_spawn_depth(Some(1)));
        let tool = SpawnAgentTool::new(host);
        // Parent already at depth 0 → child would be depth 1 → rejected when max=1.
        let mut extras = std::collections::HashMap::new();
        extras.insert(ovo_tools::EXTRA_SPAWN_DEPTH.to_owned(), "0".into());
        let ctx = ToolCallContext::default().with_extras(extras);
        let err = tool
            .call(ctx, json!({"prompt": "too deep"}))
            .await
            .expect_err("depth");
        assert_eq!(err.code(), ErrorCode::ToolExecution);
        assert!(
            err.message().contains("depth") || err.message().contains("spawn"),
            "unexpected message: {}",
            err.message()
        );
    }

    #[tokio::test]
    async fn spawn_tool_agent_type_allowlist() {
        let sampler = Arc::new(MockSampler::new());
        sampler.map_user_text("task", "ok");
        let host: Arc<dyn SessionHost> = Arc::new(InProcessHost::new(sampler, Vec::new()));
        let tool = SpawnAgentTool::new(host).with_allowed_agent_types(["explore"]);
        let err = tool
            .call(
                ToolCallContext::default(),
                json!({"prompt": "task", "agent_type": "plan"}),
            )
            .await
            .expect_err("allowlist");
        assert_eq!(err.code(), ErrorCode::ToolInvalidArgs);
        let ok = tool
            .call(
                ToolCallContext::default(),
                json!({"prompt": "task", "agent_type": "explore"}),
            )
            .await
            .expect("allowed");
        assert!(!ok.is_error);
    }

    #[tokio::test]
    async fn spawn_tool_capability_mode_fail_closed() {
        let sampler = Arc::new(MockSampler::new());
        let host: Arc<dyn SessionHost> = Arc::new(InProcessHost::new(sampler, Vec::new()));
        let tool = SpawnAgentTool::new(host);
        let err = tool
            .call(
                ToolCallContext::default(),
                json!({"prompt": "x", "capability_mode": "admin"}),
            )
            .await
            .expect_err("unknown mode");
        assert_eq!(err.code(), ErrorCode::ToolInvalidArgs);
        assert!(err.message().contains("capability_mode"));
    }
}
