//! Process-level sandbox policy and backends for tool command wrapping.
//!
//! OS backends are feature-gated. This is not a micro-VM: it wraps
//! [`tokio::process::Command`] with host-kernel policies.

#![forbid(unsafe_code)]
#![cfg_attr(
    not(all(feature = "landlock", target_os = "linux")),
    allow(
        unused_crate_dependencies,
        reason = "serde_json is for Landlock wrap JSON policy; unused when wrap is cfg'd out"
    )
)]

#[cfg(all(feature = "landlock", target_os = "linux"))]
mod landlock;
#[cfg(target_os = "linux")]
mod landlock_apply;
#[cfg(all(feature = "seatbelt", target_os = "macos"))]
mod seatbelt;

use std::ffi::OsString;
use std::path::{Path, PathBuf};

#[cfg(all(feature = "landlock", target_os = "linux"))]
pub use landlock::LandlockBackend;
#[cfg(target_os = "linux")]
#[doc(hidden)]
pub use landlock_apply::apply_landlock_policy;
#[cfg(all(feature = "seatbelt", target_os = "macos"))]
pub use seatbelt::{SANDBOX_EXEC, SeatbeltBackend, build_profile};
use serde::{Deserialize, Serialize};
use tokio::process::Command;

/// Filesystem access granted to a sandboxed command.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FsPolicy {
    /// No filesystem access beyond what the OS grants by default (backend-defined).
    None,
    /// Read-only paths (and their trees).
    ReadOnly {
        /// Absolute paths allowed for read.
        paths: Vec<PathBuf>,
    },
    /// Read-write paths (and their trees).
    ReadWrite {
        /// Absolute paths allowed for read/write.
        paths: Vec<PathBuf>,
    },
}

/// Network access granted to a sandboxed command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum NetPolicy {
    /// No network (preferred default for shell tools).
    #[default]
    Denied,
    /// Network allowed (backend may still apply finer filters).
    Allowed,
}

/// Declarative sandbox policy applied by a [`SandboxBackend`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxPolicy {
    /// Filesystem policy.
    pub fs: FsPolicy,
    /// Network policy.
    pub net: NetPolicy,
}

impl SandboxPolicy {
    /// Read-write under `root`, network denied — typical workspace shell.
    #[must_use]
    pub fn workspace(root: impl Into<PathBuf>) -> Self {
        Self {
            fs: FsPolicy::ReadWrite {
                paths: vec![root.into()],
            },
            net: NetPolicy::Denied,
        }
    }
}

/// Errors from sandbox backends.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SandboxError {
    /// Backend refused to wrap the command under the given policy.
    #[error("sandbox denied: {0}")]
    Denied(String),
    /// Backend misconfiguration or OS failure.
    #[error("sandbox failed: {0}")]
    Failed(String),
}

/// Wraps a process command so the OS enforces [`SandboxPolicy`].
pub trait SandboxBackend: Send + Sync {
    /// Stable backend id (`no_sandbox`, `seatbelt`, `landlock`, …).
    fn name(&self) -> &'static str;

    /// Apply `policy` to `cmd` (may rewrite argv/env or wrap with a helper).
    ///
    /// # Errors
    ///
    /// Returns [`SandboxError`] when the policy cannot be applied.
    fn wrap(&self, policy: &SandboxPolicy, cmd: Command) -> Result<Command, SandboxError>;
}

/// Non-enforcing backend: argv, env, and stdio are left unchanged;
/// `kill_on_drop(true)` is always applied.
///
/// Prefer [`TrustedExecution`] on tools when opting out of process sandboxing.
/// Use this when a [`SandboxBackend`] value is required without enforcement.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoSandbox;

impl SandboxBackend for NoSandbox {
    fn name(&self) -> &'static str {
        "no_sandbox"
    }

    fn wrap(&self, _policy: &SandboxPolicy, mut cmd: Command) -> Result<Command, SandboxError> {
        cmd.kill_on_drop(true);
        Ok(cmd)
    }
}

/// Marker type: the host opts out of process sandboxing for a tool.
///
/// Tool constructors take this marker so opt-out is explicit at the call site.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TrustedExecution;

#[derive(Debug)]
pub(crate) struct CommandSpec {
    pub program: OsString,
    pub args: Vec<OsString>,
    pub cwd: Option<PathBuf>,
    pub envs: Vec<(OsString, OsString)>,
}

pub(crate) fn take_command_spec(cmd: &Command) -> CommandSpec {
    let std = cmd.as_std();
    CommandSpec {
        program: std.get_program().to_os_string(),
        args: std.get_args().map(std::ffi::OsStr::to_os_string).collect(),
        cwd: std.get_current_dir().map(Path::to_path_buf),
        envs: std
            .get_envs()
            .filter_map(|(k, v)| v.map(|val| (k.to_os_string(), val.to_os_string())))
            .collect(),
    }
}

pub(crate) fn require_absolute(p: &Path) -> Result<PathBuf, SandboxError> {
    if p.is_absolute() {
        Ok(p.to_path_buf())
    } else {
        Err(SandboxError::Denied(format!(
            "sandbox path must be absolute: {}",
            p.display()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_policy_serde() {
        let p = SandboxPolicy::workspace("/tmp/ws");
        let raw = serde_json::to_string(&p).expect("ser");
        let back: SandboxPolicy = serde_json::from_str(&raw).expect("de");
        assert_eq!(back.net, NetPolicy::Denied);
        assert!(matches!(back.fs, FsPolicy::ReadWrite { .. }));
    }

    #[test]
    fn no_sandbox_passthrough() {
        let cmd = Command::new("echo");
        let out = NoSandbox.wrap(&SandboxPolicy::workspace("/tmp"), cmd);
        assert!(out.is_ok());
        assert_eq!(NoSandbox.name(), "no_sandbox");
    }

    #[test]
    fn require_absolute_rejects_relative() {
        let got = require_absolute(Path::new("/tmp")).expect("abs");
        assert_eq!(got, Path::new("/tmp"));
        let err = require_absolute(Path::new("relative")).expect_err("rel");
        assert!(matches!(err, SandboxError::Denied(_)), "{err:?}");
    }
}
