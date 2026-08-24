//! Linux Landlock backend wrapping commands with the `ovo-landlock` helper.

use std::ffi::OsString;
use std::path::PathBuf;

use tokio::process::Command;

use crate::{
    FsPolicy, SandboxBackend, SandboxError, SandboxPolicy, require_absolute, take_command_spec,
};

/// Linux Landlock backend: prefixes commands with the `ovo-landlock` helper.
#[derive(Debug, Clone)]
pub struct LandlockBackend {
    helper: PathBuf,
}

impl LandlockBackend {
    /// Discover the `ovo-landlock` helper for production use.
    ///
    /// Unit tests must not call this; use [`Self::with_helper`] instead.
    ///
    /// `cargo add ovo` does not install the helper. The host must provide it via
    /// one of: the same directory as the host binary, the `OVO_LANDLOCK_HELPER`
    /// environment variable, or
    /// `cargo install ovo-sandbox --features landlock --bin ovo-landlock`.
    ///
    /// # Errors
    ///
    /// Returns [`SandboxError::Failed`] when the helper cannot be resolved.
    pub fn new() -> Result<Self, SandboxError> {
        let helper = resolve_helper(
            std::env::var_os("OVO_LANDLOCK_HELPER").map(PathBuf::from),
            std::env::var_os("PATH"),
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(std::path::Path::to_path_buf)),
        )?;
        Ok(Self { helper })
    }

    /// Wrap commands with an explicit helper path.
    #[must_use]
    pub fn with_helper(path: impl Into<PathBuf>) -> Self {
        Self {
            helper: path.into(),
        }
    }
}

pub(crate) fn resolve_helper(
    env_helper: Option<PathBuf>,
    path_var: Option<OsString>,
    exe_dir: Option<PathBuf>,
) -> Result<PathBuf, SandboxError> {
    if let Some(path) = env_helper {
        if !path.as_os_str().is_empty() && path.is_file() {
            return Ok(path);
        }
        return Err(SandboxError::Failed(format!(
            "OVO_LANDLOCK_HELPER is not a file: {}",
            path.display()
        )));
    }
    if let Some(path_var) = path_var {
        for dir in std::env::split_paths(&path_var) {
            let candidate = dir.join("ovo-landlock");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    if let Some(exe_dir) = exe_dir {
        let candidate = exe_dir.join("ovo-landlock");
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(SandboxError::Failed("ovo-landlock helper not found".into()))
}

impl SandboxBackend for LandlockBackend {
    fn name(&self) -> &'static str {
        "landlock"
    }

    fn wrap(&self, policy: &SandboxPolicy, cmd: Command) -> Result<Command, SandboxError> {
        match &policy.fs {
            FsPolicy::None => {
                return Err(SandboxError::Denied("landlock fs policy none".into()));
            }
            FsPolicy::ReadOnly { paths } | FsPolicy::ReadWrite { paths } => {
                for path in paths {
                    require_absolute(path)?;
                }
            }
        }
        let json =
            serde_json::to_string(policy).map_err(|e| SandboxError::Failed(e.to_string()))?;
        if !self.helper.is_file() {
            return Err(SandboxError::Failed(format!(
                "ovo-landlock helper not found: {}",
                self.helper.display()
            )));
        }
        let spec = take_command_spec(&cmd);
        let mut wrapped = Command::new(&self.helper);
        wrapped.arg("-p").arg(json);
        wrapped.arg(&spec.program);
        wrapped.args(spec.args);
        if let Some(dir) = spec.cwd {
            wrapped.current_dir(dir);
        }
        for (key, val) in spec.envs {
            wrapped.env(key, val);
        }
        wrapped.kill_on_drop(true);
        Ok(wrapped)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "unit tests")]
mod tests {
    use std::ffi::OsStr;
    use std::path::PathBuf;

    use tempfile::tempdir;

    use super::*;
    use crate::{FsPolicy, NetPolicy, SandboxPolicy};

    fn fs_rw_net_allowed(root: PathBuf) -> SandboxPolicy {
        SandboxPolicy {
            fs: FsPolicy::ReadWrite { paths: vec![root] },
            net: NetPolicy::Allowed,
        }
    }

    #[test]
    fn with_helper_wrap_prefixes_argv() {
        let dir = tempdir().expect("temp");
        let helper = dir.path().join("ovo-landlock");
        std::fs::write(&helper, b"").expect("write");
        let backend = LandlockBackend::with_helper(&helper);
        let policy = SandboxPolicy::workspace("/tmp/ws");
        let cmd = Command::new("/bin/echo");
        let wrapped = backend.wrap(&policy, cmd).expect("wrap");
        assert_eq!(wrapped.as_std().get_program(), helper.as_os_str());
        let args: Vec<_> = wrapped.as_std().get_args().collect();
        let flag = args.first().expect("flag");
        let json = args.get(1).expect("json");
        let prog = args.get(2).expect("program");
        assert_eq!(*flag, OsStr::new("-p"));
        let parsed: SandboxPolicy =
            serde_json::from_str(&json.to_string_lossy()).expect("policy json");
        assert_eq!(parsed, policy);
        assert_eq!(*prog, OsStr::new("/bin/echo"));
    }

    #[test]
    fn wrap_rejects_relative_paths() {
        let backend = LandlockBackend::with_helper("/nonexistent");
        let policy = fs_rw_net_allowed(PathBuf::from("relative"));
        let err = backend
            .wrap(&policy, Command::new("echo"))
            .expect_err("relative");
        assert!(matches!(err, SandboxError::Denied(_)), "{err:?}");
    }

    #[test]
    fn wrap_rejects_fs_none() {
        let backend = LandlockBackend::with_helper("/nonexistent");
        let policy = SandboxPolicy {
            fs: FsPolicy::None,
            net: NetPolicy::Allowed,
        };
        let err = backend
            .wrap(&policy, Command::new("echo"))
            .expect_err("none");
        assert!(matches!(err, SandboxError::Denied(_)), "{err:?}");
    }

    #[test]
    fn resolve_helper_env_wins() {
        let dir = tempdir().expect("temp");
        let helper = dir.path().join("ovo-landlock");
        std::fs::write(&helper, b"").expect("write");
        let got = resolve_helper(Some(helper.clone()), None, None).expect("resolve");
        assert_eq!(got, helper);
    }

    #[test]
    fn resolve_helper_env_missing_is_failed() {
        let err = resolve_helper(
            Some(PathBuf::from("/nonexistent/ovo-landlock")),
            Some(OsString::from("/bin")),
            Some(PathBuf::from("/usr/bin")),
        )
        .expect_err("missing");
        assert!(matches!(err, SandboxError::Failed(_)), "{err:?}");
    }

    #[test]
    fn resolve_helper_exe_dir() {
        let dir = tempdir().expect("temp");
        let helper = dir.path().join("ovo-landlock");
        std::fs::write(&helper, b"").expect("write");
        let got = resolve_helper(None, None, Some(dir.path().to_path_buf())).expect("resolve");
        assert_eq!(got, helper);
    }
}
