//! Offline turn with `MockSampler` (no network).
//!
//! 非 S-plane：AutoApprove（无 Destructive 工具）。生产用 `TurnOptions::for_host` + `sandboxed_host`.
#![allow(
    clippy::print_stdout,
    clippy::expect_used,
    unused_crate_dependencies,
    reason = "offline demo binary uses stdout and expect for setup"
)]

use std::sync::Arc;

use ovo::{AgentBuilder, MockSampler, TurnInput, TurnOptions, TurnRuntime, VecConversationState};

#[tokio::main]
async fn main() {
    let sampler = Arc::new(MockSampler::new());
    sampler.push_text("hello from mock");

    let agent = AgentBuilder::named("assistant")
        .instructions("You are helpful.")
        .model("mock")
        .build()
        .expect("build agent");

    let mut state = VecConversationState::new();
    let runtime = TurnRuntime::new();
    let outcome = runtime
        .run(
            &agent,
            sampler.as_ref(),
            &mut state,
            TurnInput::Text("hi".into()),
            TurnOptions::default(),
        )
        .await
        .expect("turn");

    println!("output: {}", outcome.output_text);
    println!("steps: {}", outcome.steps);
}
