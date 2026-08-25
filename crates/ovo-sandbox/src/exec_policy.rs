//! Argv allowlist. Not a sandbox.

use std::sync::Arc;

/// Decision for a candidate argv.
///
/// Not `non_exhaustive`: W8 has two outcomes (wrap or `ToolDenied`) and
/// toolkit crates must match exhaustively without a `_` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExecDecision {
    /// Enter `SandboxBackend::wrap`. Does **not** skip `ApprovalPolicy::Destructive`.
    Allow,
    /// Do not wrap; tool returns `ErrorCode::ToolDenied`.
    Deny,
}

/// Argv policy consulted inside the tool, before wrap.
pub trait ExecPolicy: Send + Sync + std::fmt::Debug {
    /// Decide for a tokenized argv (`sh -c` tokens or `ExecSession` JSON array).
    fn decide(&self, argv: &[String]) -> ExecDecision;
}

/// Most severe match wins: **Deny > Allow**. Unmatched → Deny.
#[derive(Debug, Clone)]
pub struct PrefixExecPolicy {
    rules: Vec<PrefixRule>,
}

/// Exact prefix of argv tokens.
#[derive(Debug, Clone)]
pub struct PrefixRule {
    /// e.g. `["git", "status"]`. Empty tokens never match. Bare `["git"]` is forbidden in
    /// [`PrefixExecPolicy::workspace_shell`].
    pub tokens: Vec<String>,
    /// Decision when this prefix matches.
    pub decision: ExecDecision,
}

impl PrefixExecPolicy {
    /// Build from rules. Empty `tokens` are ignored (do not match-all).
    #[must_use]
    pub fn new(rules: Vec<PrefixRule>) -> Self {
        Self { rules }
    }

    /// Production allowlist used by `default_toolkit`.
    #[must_use]
    pub fn workspace_shell() -> Self {
        Self::new(vec![
            PrefixRule {
                tokens: vec!["ls".into()],
                decision: ExecDecision::Allow,
            },
            PrefixRule {
                tokens: vec!["pwd".into()],
                decision: ExecDecision::Allow,
            },
            PrefixRule {
                tokens: vec!["cat".into()],
                decision: ExecDecision::Allow,
            },
            PrefixRule {
                tokens: vec!["git".into(), "status".into()],
                decision: ExecDecision::Allow,
            },
            PrefixRule {
                tokens: vec!["git".into(), "diff".into()],
                decision: ExecDecision::Allow,
            },
            PrefixRule {
                tokens: vec!["git".into(), "log".into()],
                decision: ExecDecision::Allow,
            },
            PrefixRule {
                tokens: vec!["sudo".into()],
                decision: ExecDecision::Deny,
            },
            PrefixRule {
                tokens: vec!["chmod".into()],
                decision: ExecDecision::Deny,
            },
            PrefixRule {
                tokens: vec!["curl".into()],
                decision: ExecDecision::Deny,
            },
            PrefixRule {
                tokens: vec!["ssh".into()],
                decision: ExecDecision::Deny,
            },
            PrefixRule {
                tokens: vec!["rm".into()],
                decision: ExecDecision::Deny,
            },
        ])
    }
}

impl ExecPolicy for PrefixExecPolicy {
    fn decide(&self, argv: &[String]) -> ExecDecision {
        let mut best: Option<ExecDecision> = None;
        for rule in &self.rules {
            if rule.tokens.is_empty() {
                continue;
            }
            if argv.len() >= rule.tokens.len() && argv.iter().zip(&rule.tokens).all(|(a, t)| a == t)
            {
                best = Some(match best {
                    None => rule.decision,
                    Some(cur) => cur.max(rule.decision),
                });
            }
        }
        best.unwrap_or(ExecDecision::Deny)
    }
}

/// Default for `ShellTool::sandboxed` / `with_no_sandbox`.
#[derive(Debug, Default, Clone, Copy)]
pub struct DenyAllExecPolicy;

impl ExecPolicy for DenyAllExecPolicy {
    fn decide(&self, _argv: &[String]) -> ExecDecision {
        ExecDecision::Deny
    }
}

/// `ShellTool::trusted(TrustedExecution)` and tests.
#[derive(Debug, Default, Clone, Copy)]
pub struct AllowAllExecPolicy;

impl ExecPolicy for AllowAllExecPolicy {
    fn decide(&self, _argv: &[String]) -> ExecDecision {
        ExecDecision::Allow
    }
}

/// Shared handle stored on tools.
pub type SharedExecPolicy = Arc<dyn ExecPolicy>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_shell_table() {
        let p = PrefixExecPolicy::workspace_shell();
        let cases: &[(&[&str], ExecDecision)] = &[
            (&["ls"], ExecDecision::Allow),
            (&["ls", "-la"], ExecDecision::Allow),
            (&["pwd"], ExecDecision::Allow),
            (&["cat", "/etc/passwd"], ExecDecision::Allow), // OS wrap is the jail
            (&["git", "status"], ExecDecision::Allow),
            (&["git", "status", "--porcelain"], ExecDecision::Allow),
            (&["git", "diff"], ExecDecision::Allow),
            (&["git", "log"], ExecDecision::Allow),
            (&["git"], ExecDecision::Deny),
            (&["git", "push"], ExecDecision::Deny),
            (&["sudo", "ls"], ExecDecision::Deny),
            (&["chmod", "777", "x"], ExecDecision::Deny),
            (&["curl", "https://example.com"], ExecDecision::Deny),
            (&["ssh", "host"], ExecDecision::Deny),
            (&["rm", "-rf", "/"], ExecDecision::Deny),
            (&["python"], ExecDecision::Deny),
            (&[], ExecDecision::Deny),
        ];
        for (argv, want) in cases {
            let got = p.decide(&argv.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>());
            assert_eq!(got, *want, "argv={argv:?}");
        }
    }

    #[test]
    fn severity_deny_beats_allow() {
        let p = PrefixExecPolicy::new(vec![
            PrefixRule {
                tokens: vec!["git".into()],
                decision: ExecDecision::Allow,
            },
            PrefixRule {
                tokens: vec!["git".into(), "push".into()],
                decision: ExecDecision::Deny,
            },
        ]);
        assert_eq!(p.decide(&["git".into(), "push".into()]), ExecDecision::Deny);
    }
}
