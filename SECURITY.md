# Security

## Supported versions

Only the `main` line (`0.9.1` workspace version) receives fixes.
Older Machi ≤0.8 is unsupported.
Historical crates.io `machi*` 1.0.0 and the `ovo` 0.9.0 stub are unsupported.

## Reporting

Report vulnerabilities privately to the maintainers via the repository’s
security advisory channel (GitHub Security Advisories) when available, or by
contacting the org listed in the repository metadata. Do not open public
issues for unfixed critical flaws.

## Dependency policy

```bash
cargo deny check
```

- **Advisories:** fail on known RustSec issues (see `deny.toml`).
- **Licenses:** explicit allow-list (MIT/Apache-2.0 family + common
  permissive deps used by TLS/ICU).
- **Bans / sources:** default deny-template rules; crates.io is the expected
  source for third-party crates.

## Runtime trust model

- Default path is **offline** (`MockSampler`); network providers are feature-gated.
- Toolkit tools are **cwd-jailed**; do not treat them as a full OS sandbox.
- cwd-jail ≠ OS sandbox ≠ ExecPolicy allowlist.
- Default isolation is **in-process** (`InProcessIsolation`). Product sandboxes
  inject a custom `IsolationBackend`.
- Depth, concurrency, and journal divergence are **always** fail-closed.
- Host `agent_budget` defaults to 128; `with_agent_budget` caps at 1024.
  Unlimited budget requires `with_unlimited_agent_budget(TrustedExecution)`.
- Workflow adapter is a **second** 128 gate (`run_workflow_on_host(None)` → 128),
  not the same pool as the host counter.
- Production path: `TurnOptions::for_host` + `sandboxed_host` /
  `default_toolkit(jail, backend)` + an installed OS backend. Linux helper:
  same directory as the host binary, `OVO_LANDLOCK_HELPER`, or PATH.
  `platform_sandbox()` can fail for a pure library dependency.
  `cargo add ovo` (default features) does **not** compile `ovo-toolkit`; it
  does compile `ovo-sandbox` (runtime → `TrustedExecution`). Features
  `toolkit` / `sandbox` still gate re-exports and `sandboxed_host`.
- `full` does not include OS backends; `--all-features` does.
- Workflow engine never loads LLM HTTP clients (`ovo-workflow` firewall).
