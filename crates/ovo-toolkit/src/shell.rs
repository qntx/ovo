//! Cwd-jailed shell command execution with explicit sandbox policy.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use ovo_sandbox::{NoSandbox, SandboxBackend, SandboxPolicy, TrustedExecution};
use ovo_tools::stream::ToolStream;
use ovo_tools::{
    DynTool, ToolCallContext, ToolError, ToolMetadata, ToolProgress, ToolResult, with_progress,
};
use serde_json::{Value, json};
use tokio::process::Command;
use tokio::time::timeout;

use crate::jail::resolve_root;

/// Default command timeout.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
/// Default combined stdout+stderr capture limit.
pub const DEFAULT_MAX_OUTPUT: usize = 64 * 1024;

/// How the shell process is isolated.
#[derive(Clone)]
enum ShellIsolation {
    /// Explicit opt-out of process sandboxing.
    Trusted,
    /// OS-enforced backend + policy.
    Sandboxed {
        backend: Arc<dyn SandboxBackend>,
        policy: SandboxPolicy,
    },
}

impl std::fmt::Debug for ShellIsolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Trusted => f.write_str("Trusted"),
            Self::Sandboxed { backend, policy } => f
                .debug_struct("Sandboxed")
                .field("backend", &backend.name())
                .field("policy", policy)
                .finish(),
        }
    }
}

/// Run a shell command with `cwd` jailed to the workspace root.
///
/// Construction **requires** an explicit isolation choice:
/// - [`ShellTool::trusted`] — host opts out of process sandbox (marker type)
/// - [`ShellTool::sandboxed`] — wrap via [`SandboxBackend`]
/// - [`ShellTool::with_no_sandbox`] — backend present but non-enforcing
///
/// There is no `Default` impl: silent trust is forbidden (secure-by-default).
#[derive(Debug, Clone)]
pub struct ShellTool {
    /// Jail / working directory.
    pub jail_root: Option<PathBuf>,
    /// Per-call timeout.
    pub timeout: Duration,
    /// Max captured output bytes.
    pub max_output: usize,
    isolation: ShellIsolation,
}

impl ShellTool {
    /// Trusted host: no process sandbox (explicit opt-out).
    #[must_use]
    pub fn trusted(root: impl Into<PathBuf>, _trust: TrustedExecution) -> Self {
        Self {
            jail_root: Some(root.into()),
            timeout: DEFAULT_TIMEOUT,
            max_output: DEFAULT_MAX_OUTPUT,
            isolation: ShellIsolation::Trusted,
        }
    }

    /// Sandboxed shell: command is wrapped by `backend` under `policy`.
    #[must_use]
    pub fn sandboxed(
        root: impl Into<PathBuf>,
        backend: Arc<dyn SandboxBackend>,
        policy: SandboxPolicy,
    ) -> Self {
        Self {
            jail_root: Some(root.into()),
            timeout: DEFAULT_TIMEOUT,
            max_output: DEFAULT_MAX_OUTPUT,
            isolation: ShellIsolation::Sandboxed { backend, policy },
        }
    }

    /// Explicit `NoSandbox` backend (same non-enforcement as trusted, but
    /// keeps a [`SandboxBackend`] name in the isolation record).
    #[must_use]
    pub fn with_no_sandbox(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self::sandboxed(
            root.clone(),
            Arc::new(NoSandbox),
            SandboxPolicy::workspace(root),
        )
    }

    /// Set timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[async_trait]
impl DynTool for ShellTool {
    fn name(&self) -> &'static str {
        "shell"
    }

    fn description(&self) -> &'static str {
        "Run a shell command with cwd set to the workspace jail. Args: command (string)."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command line (executed via `sh -c`)"
                }
            },
            "required": ["command"],
            "additionalProperties": false
        })
    }

    fn metadata(&self) -> ToolMetadata {
        ToolMetadata::shell_execute(self.timeout)
    }

    async fn call(&self, ctx: ToolCallContext, arguments: Value) -> Result<ToolResult, ToolError> {
        let stream = self.execute(ctx, arguments).await;
        ovo_tools::drain_terminal(stream).await
    }

    async fn execute(&self, ctx: ToolCallContext, arguments: Value) -> ToolStream {
        let command = arguments
            .get("command")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned);

        let root = match resolve_root(self.jail_root.as_ref(), &ctx, "shell") {
            Ok(r) => r,
            Err(e) => return ovo_tools::terminal_only(Err(e)),
        };
        let Some(command) = command else {
            return ovo_tools::terminal_only(Err(ovo_tools::error::codes::invalid_args(
                "shell requires non-empty command",
            )));
        };
        let limit = self.timeout;
        let max_output = self.max_output;
        let cancel = ctx.cancel.clone();
        let isolation = self.isolation.clone();
        with_progress(
            vec![ToolProgress::text(format!("shell: {command}"))],
            move || async move {
                if cancel.is_cancelled() {
                    return Err(ovo_tools::error::codes::cancelled());
                }
                let mut cmd = Command::new("sh");
                cmd.arg("-c").arg(&command).current_dir(&root);

                let mut cmd = match &isolation {
                    ShellIsolation::Trusted => cmd,
                    ShellIsolation::Sandboxed { backend, policy } => {
                        backend.wrap(policy, cmd).map_err(|e| {
                            ovo_tools::error::codes::execution(format!(
                                "sandbox '{}': {e}",
                                backend.name()
                            ))
                        })?
                    }
                };
                cmd.stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .kill_on_drop(true);

                let child = cmd
                    .spawn()
                    .map_err(|e| ovo_tools::error::codes::execution(format!("spawn shell: {e}")))?;

                let wait = child.wait_with_output();
                let output = tokio::select! {
                    () = cancel.cancelled() => {
                        return Err(ovo_tools::error::codes::cancelled());
                    }
                    res = timeout(limit, wait) => {
                        match res {
                            Ok(Ok(o)) => o,
                            Ok(Err(e)) => {
                                return Err(ovo_tools::error::codes::execution(format!(
                                    "shell wait: {e}"
                                )));
                            }
                            Err(_) => {
                                return Err(ovo_tools::error::codes::timeout(format!(
                                    "shell timed out after {limit:?}"
                                )));
                            }
                        }
                    }
                };

                let mut stdout = String::from_utf8_lossy(&output.stdout).into_owned();
                let mut stderr = String::from_utf8_lossy(&output.stderr).into_owned();
                truncate_in_place(&mut stdout, max_output / 2);
                truncate_in_place(&mut stderr, max_output / 2);
                let code = output.status.code().unwrap_or(-1);
                let mut content = format!("exit_code={code}\n");
                if !stdout.is_empty() {
                    content.push_str("--- stdout ---\n");
                    content.push_str(&stdout);
                    if !content.ends_with('\n') {
                        content.push('\n');
                    }
                }
                if !stderr.is_empty() {
                    content.push_str("--- stderr ---\n");
                    content.push_str(&stderr);
                }
                Ok(ToolResult {
                    content,
                    structured: Some(json!({
                        "exit_code": code,
                        "stdout": stdout,
                        "stderr": stderr,
                        "command": command,
                    })),
                    is_error: !output.status.success(),
                })
            },
        )
    }
}

fn truncate_in_place(s: &mut String, max: usize) {
    if s.len() > max {
        s.truncate(max);
        s.push_str("\n…[truncated]");
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "unit tests")]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[tokio::test]
    async fn echoes_in_jail_trusted() {
        let dir = tempdir().expect("temp");
        let tool = ShellTool::trusted(dir.path(), TrustedExecution);
        let r = tool
            .call(
                ToolCallContext {
                    cwd: Some(dir.path().to_path_buf()),
                    ..ToolCallContext::default()
                },
                json!({"command": "echo hi && pwd"}),
            )
            .await
            .expect("shell");
        assert!(r.content.contains("hi"), "{}", r.content);
    }

    #[tokio::test]
    async fn echoes_with_no_sandbox_backend() {
        let dir = tempdir().expect("temp");
        let tool = ShellTool::with_no_sandbox(dir.path());
        let r = tool
            .call(
                ToolCallContext {
                    cwd: Some(dir.path().to_path_buf()),
                    ..ToolCallContext::default()
                },
                json!({"command": "echo sandboxed-slot"}),
            )
            .await
            .expect("shell");
        assert!(r.content.contains("sandboxed-slot"), "{}", r.content);
    }

    #[tokio::test]
    #[cfg(all(feature = "seatbelt", target_os = "macos"))]
    async fn seatbelt_shell_allows_inside_blocks_outside() {
        use std::sync::Arc;

        use ovo_sandbox::{SandboxPolicy, SeatbeltBackend};

        let dir = tempdir().expect("temp");
        let root = dir.path().canonicalize().expect("canon");
        std::fs::write(root.join("in.txt"), b"inside-ok").expect("in");

        let outside = std::env::var_os("HOME")
            .map(PathBuf::from)
            .expect("HOME")
            .join(format!("ovo_shell_out_{}", std::process::id()));
        std::fs::write(&outside, b"secret").expect("out");

        let tool = ShellTool::sandboxed(
            root.clone(),
            Arc::new(SeatbeltBackend),
            SandboxPolicy::workspace(root.clone()),
        );

        let inside = tool
            .call(
                ToolCallContext {
                    cwd: Some(root.clone()),
                    ..ToolCallContext::default()
                },
                json!({"command": format!("cat {}", root.join("in.txt").display())}),
            )
            .await
            .expect("inside");
        assert!(
            !inside.is_error && inside.content.contains("inside-ok"),
            "{}",
            inside.content
        );

        let out = tool
            .call(
                ToolCallContext {
                    cwd: Some(root),
                    ..ToolCallContext::default()
                },
                json!({"command": format!("cat {}", outside.display())}),
            )
            .await
            .expect("outside call");
        assert!(
            out.is_error,
            "outside read must fail under seatbelt: {}",
            out.content
        );
        let _ = std::fs::remove_file(&outside);
    }

    #[tokio::test]
    #[cfg(all(feature = "landlock", target_os = "linux"))]
    async fn landlock_shell_allows_inside_blocks_outside() {
        use std::sync::Arc;

        use ovo_sandbox::{FsPolicy, LandlockBackend, NetPolicy, SandboxPolicy};

        let Some(helper) = std::env::var_os("OVO_LANDLOCK_HELPER") else {
            return;
        };

        let dir = tempdir().expect("temp");
        let root = dir.path().canonicalize().expect("canon");
        std::fs::write(root.join("in.txt"), b"inside-ok").expect("in");

        let outside = std::env::var_os("HOME")
            .map(PathBuf::from)
            .expect("HOME")
            .join(format!("ovo_shell_ll_out_{}", std::process::id()));
        std::fs::write(&outside, b"secret").expect("out");

        let tool = ShellTool::sandboxed(
            root.clone(),
            Arc::new(LandlockBackend::with_helper(helper)),
            SandboxPolicy {
                fs: FsPolicy::ReadWrite {
                    paths: vec![root.clone()],
                },
                net: NetPolicy::Allowed,
            },
        );

        let inside = tool
            .call(
                ToolCallContext {
                    cwd: Some(root.clone()),
                    ..ToolCallContext::default()
                },
                json!({"command": format!("cat {}", root.join("in.txt").display())}),
            )
            .await
            .expect("inside");
        assert!(
            !inside.is_error && inside.content.contains("inside-ok"),
            "{}",
            inside.content
        );

        let out = tool
            .call(
                ToolCallContext {
                    cwd: Some(root),
                    ..ToolCallContext::default()
                },
                json!({"command": format!("cat {}", outside.display())}),
            )
            .await
            .expect("outside call");
        assert!(
            out.is_error,
            "outside read must fail under landlock: {}",
            out.content
        );
        let _ = std::fs::remove_file(&outside);
    }
}
