# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

**Stability:** workspace crates are **`0.9.x` pre-stability**. The crates.io
publish of `1.0.0` was premature and is **not** an API freeze. Breaking changes
land without compatibility layers ([`AGENTS.md`](./AGENTS.md)). A real `1.0` requires the product charter planes closed.

## [Unreleased]

### Breaking

- `default_toolkit(jail, backend) -> Result<_, SandboxError>`; trusted shell is
  `trusted_toolkit(..., TrustedExecution)`. No one-arg alias.
- `ShellTool::sandboxed` defaults to `DenyAllExecPolicy`.
- Sandbox `wrap` does not set stdio (`kill_on_drop(true)` only).
- `InProcessHost::new`: agent budget 128, approval `AlwaysDeny`; `spawn_one`
  copies host approval and `ApprovalPolicy::Destructive`.
- `run_workflow_on_host(None)` (and configured variants) applies budget 128;
  `None` is not unlimited.
- `ovo-runtime` depends on `ovo-sandbox` for `TrustedExecution`. It does **not**
  depend on `ovo-toolkit`. Production factory is facade `sandboxed_host`.

### Added

- Feature `landlock`, `LandlockBackend`, helper bin `ovo-landlock`.
- `ExecPolicy` family (`PrefixExecPolicy`, `DenyAllExecPolicy`,
  `AllowAllExecPolicy`, `workspace_shell()`).
- `TurnOptions::for_host`, facade `sandboxed_host`, `platform_sandbox`,
  `ChildToolkit`.
- CI job `sandbox-macos` (`cargo test -p ovo-sandbox -p ovo-toolkit
  --features seatbelt --all-targets`).
- README honesty: root Embed section and crates.io page (production
  constructors, feature boundaries, Linux helper contract).

## [0.9.1] — 2026-08-21

Clean-break product-prefix rename. No compatibility layer, dual headers,
directory fallbacks, or `machi` re-exports.

### Breaking

- **Crates:** `machi` / `machi-*` → `ovo` / `ovo-*` (13 workspace crates).
  Downstream `use machi::…` fails. Historical crates.io `machi*` 1.0.0 is
  frozen and not yanked.
- **Types:** `MachiError` → `OvoError`.
- **Durable headers:** `# ovo-journal/2`, `# ovo-session-events/1`,
  `__ovo_host_error`. Session event load rejects unknown `#` headers
  (same gate as journal). Old `# machi-*` files fail-closed. Append does
  not rewrite existing headers.
- **Default dirs:** `.ovo/sessions`, `.ovo/agents`, `~/.ovo/agents`.
- **Metrics / traces:** `ovo_*` series, `ovo.*` spans/fields. No `machi_*`
  dual-write.
- **Env / extras:** `OVO_OLLAMA_MODEL`; tool extras `ovo.spawn_depth`;
  OpenAI schema name `ovo_output`.

### Added

- **`ovo-sandbox` (W8.2 port):** `SandboxPolicy` / `FsPolicy` / `NetPolicy`,
  `SandboxBackend`, `NoSandbox`, `TrustedExecution` marker.
- **`SeatbeltBackend` (W8.2b, feature `seatbelt`, macOS):** wraps commands with
  `/usr/bin/sandbox-exec`; workspace read-write + network deny by default;
  outside-jail reads fail.
- **Shell secure-by-default (partial W8.5):** `ShellTool` has **no** `Default`;
  construct via `trusted(TrustedExecution)`, `sandboxed(backend, policy)`, or
  `with_no_sandbox`. `default_toolkit` uses explicit trusted shell.
- **`TurnEvent` live observation surface (W7):**
  - `ovo-protocol::{TurnEvent, TurnEventKind}` (serde, `non_exhaustive`)
  - `EventBus` / `EventSink`; `TurnOptions::with_event_tx` / `with_events`
  - Turn loop emits started/step/tools/compaction/stationarity/finished/aborted
  - Stream path forwards `TextDelta` / `ReasoningDelta`
  - Nested spawn emits `SpawnStarted` / `SpawnFinished` on the parent bus
  - Mode B: `run_workflow_configured_with_events` wires the same spawn events
  - Example: `cargo run -p ovo --example live_events --features runtime`
- **Jail realpath (W8.1):** `resolve_jailed` canonicalizes deepest existing ancestor;
  in-jail symlink → outside host path is rejected (`EscapesJail`).

### Changed

- **Version:** workspace package version **`0.9.1`** (next publishable `ovo`
  after the crates.io 0.9.0 stub).
- **Docs:** remove internal `ROADMAP.md` from the repository; slim README and
  crate docs (drop phase tags, maturity banners, process narrative).
- **Dependencies:** bump workspace crates to current crates.io releases
  (`jsonschema` 0.28→0.49, `sha2` 0.10→0.11, `rhai` 1.25, `tokio` 1.53,
  `reqwest` 0.13.4, …); pin `tempfile`/`libc` via workspace deps.
- **Control plane fail-closed:**
  - Tool dispatch: per-call cancel child; **timeout cancels** nested work.
  - Host: **refund** agent budget on pre-start failures (isolation/build/state/resume err).
  - Completion gate: exhausted reminders → `GateDecision::Fail` → `ErrorCode::RuntimeGate`
    (no silent success).
- **Errors:** remove `OvoError::cancelled` alias; use `runtime_cancelled` /
  `llm_cancelled` by domain.
- **Publish:** `publish.yml` lists `ovo-sandbox` immediately before
  `ovo-toolkit` and passes reusable-workflow `timeout-minutes: 60`.

### Prior (historical)

See git history for W1–W6 work while the tree was mis-labeled `1.0.0`.

## [0.9.0] — 2026-08-12

Pre-stability line after correcting the premature `1.0.0` label.

[Unreleased]: https://github.com/qntx/ovo/compare/v0.9.1...HEAD
[0.9.1]: https://github.com/qntx/ovo/releases/tag/v0.9.1
[0.9.0]: https://github.com/qntx/ovo/releases/tag/v0.9.0
