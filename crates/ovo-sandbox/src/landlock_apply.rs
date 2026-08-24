//! Apply a [`SandboxPolicy`] via the `landlock` crate (Linux only).

use std::path::PathBuf;

use landlock::{
    ABI, Access, AccessFs, AccessNet, BitFlags, CompatLevel, Compatible, PathBeneath, PathFd,
    Ruleset, RulesetAttr, RulesetCreated, RulesetCreatedAttr, RulesetStatus, path_beneath_rules,
};

use crate::{FsPolicy, NetPolicy, SandboxError, SandboxPolicy, require_absolute};

const SYSTEM_RO: &[&str] = &["/usr", "/bin", "/lib", "/lib64", "/etc", "/dev"];

/// Apply Landlock to the **current** thread, then return so the helper can `exec`.
///
/// Only the `ovo-landlock` helper binary should call this. Applying in-process
/// and continuing the kernel in the same thread is not a supported embedding.
///
/// # Errors
///
/// Returns [`SandboxError::Denied`] for `FsPolicy::None` or relative paths, and
/// [`SandboxError::Failed`] when the kernel cannot enforce the ruleset.
#[doc(hidden)]
pub fn apply_landlock_policy(policy: &SandboxPolicy) -> Result<(), SandboxError> {
    let (ro, rw) = policy_fs_paths(policy)?;
    let abi = ABI::V1;
    let mut ruleset = Ruleset::default()
        .set_compatibility(CompatLevel::HardRequirement)
        .handle_access(AccessFs::from_all(abi))
        .map_err(ll_failed)?;
    if matches!(policy.net, NetPolicy::Denied) {
        ruleset = ruleset
            .handle_access(AccessNet::BindTcp)
            .map_err(ll_failed)?;
        ruleset = ruleset
            .handle_access(AccessNet::ConnectTcp)
            .map_err(ll_failed)?;
    }
    let mut created = ruleset.create().map_err(ll_failed)?;
    created = created
        .add_rules(path_beneath_rules(SYSTEM_RO, AccessFs::from_read(abi)))
        .map_err(ll_failed)?;
    created = add_policy_paths(created, &ro, AccessFs::from_read(abi))?;
    created = add_policy_paths(created, &rw, AccessFs::from_all(abi))?;
    let status = created.restrict_self().map_err(ll_failed)?;
    if matches!(status.ruleset, RulesetStatus::NotEnforced) {
        return Err(SandboxError::Failed("landlock ruleset not enforced".into()));
    }
    Ok(())
}

fn policy_fs_paths(policy: &SandboxPolicy) -> Result<(Vec<PathBuf>, Vec<PathBuf>), SandboxError> {
    match &policy.fs {
        FsPolicy::None => Err(SandboxError::Denied("landlock fs policy none".into())),
        FsPolicy::ReadOnly { paths } => Ok((absolute_paths(paths)?, Vec::new())),
        FsPolicy::ReadWrite { paths } => Ok((Vec::new(), absolute_paths(paths)?)),
    }
}

fn absolute_paths(paths: &[PathBuf]) -> Result<Vec<PathBuf>, SandboxError> {
    paths.iter().map(|p| require_absolute(p)).collect()
}

fn add_policy_paths(
    mut created: RulesetCreated,
    paths: &[PathBuf],
    access: BitFlags<AccessFs>,
) -> Result<RulesetCreated, SandboxError> {
    for path in paths {
        let fd = PathFd::new(path).map_err(|e| {
            SandboxError::Failed(format!(
                "sandbox path not openable: {}: {e}",
                path.display()
            ))
        })?;
        created = created
            .add_rule(PathBeneath::new(fd, access))
            .map_err(ll_failed)?;
    }
    Ok(created)
}

fn ll_failed(err: impl std::fmt::Display) -> SandboxError {
    SandboxError::Failed(err.to_string())
}
