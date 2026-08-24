//! Session source-of-truth + file checkpoint: run, drop handle, reopen, continue.
//!
//! 非 S-plane：AutoApprove（无 Destructive 工具）。生产用 `TurnOptions::for_host` + `sandboxed_host`.
//!
//! ```bash
//! cargo run -p ovo --example session_resume --features state
//! ```
#![allow(
    clippy::print_stdout,
    clippy::expect_used,
    unused_crate_dependencies,
    reason = "demo binary"
)]

use std::sync::Arc;

use ovo::{
    AgentBuilder, ChatStateHandle, FilePersistence, MockSampler, NoopMetrics, Session, TurnInput,
    TurnOptions,
};
use tempfile::tempdir;

#[tokio::main]
async fn main() {
    let dir = tempdir().expect("tmp");
    let store = FilePersistence::for_session(dir.path(), "demo_sess");
    println!("checkpoint={}", store.path().display());

    let sampler = Arc::new(MockSampler::new());
    sampler.push_text("first-turn");
    sampler.push_text("second-turn");

    let agent = AgentBuilder::named("assistant")
        .instructions("Be brief.")
        .model("mock")
        .build()
        .expect("agent");

    let mut session = Session::new();

    // Turn 1: create + checkpoint.
    let (h1, o1) = session
        .open_checkpointed_turn(
            &agent,
            sampler.as_ref(),
            &store,
            TurnInput::Text("hi".into()),
            TurnOptions::default(),
            &NoopMetrics,
        )
        .await
        .expect("turn1");
    println!("turn1={}", o1.output_text);
    assert_eq!(o1.output_text, "first-turn", "first checkpointed turn");
    let n1 = h1.messages().await.expect("msgs1").len();
    h1.shutdown().await;

    // Simulate process restart: new handle from disk.
    assert!(store.exists(), "checkpoint file must exist after turn1");
    let h2 = ChatStateHandle::open_or_new(&store).await.expect("reload");
    let n2 = h2.messages().await.expect("msgs2").len();
    assert_eq!(n1, n2, "reload must restore message history");
    println!("reloaded_messages={n2}");

    // Turn 2 on restored handle + checkpoint.
    let o2 = session
        .run_turn_on_handle_checkpointed(
            &agent,
            sampler.as_ref(),
            &h2,
            TurnInput::Text("again".into()),
            TurnOptions::default(),
            &NoopMetrics,
            Some(&store),
        )
        .await
        .expect("turn2");
    println!("turn2={}", o2.output_text);
    assert_eq!(o2.output_text, "second-turn", "second turn after resume");
    assert!(
        h2.messages().await.expect("msgs3").len() > n2,
        "history must grow after second turn"
    );

    h2.shutdown().await;
    println!("session_resume_ok=true");
}
