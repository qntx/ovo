//! Session source-of-truth: `ChatStateHandle` + optional file checkpoint.
//!
//! Canonical multi-turn path (not the turn-local `VecConversationState` alone).
//!
//! 非 S-plane：AutoApprove（无 Destructive 工具）。生产用 `TurnOptions::for_host` + `sandboxed_host`.
#![allow(
    clippy::print_stdout,
    clippy::expect_used,
    unused_crate_dependencies,
    reason = "offline demo binary uses stdout and expect for setup"
)]

use std::sync::Arc;

use ovo::{
    AgentBuilder, ChatPersistence, FilePersistence, MemoryPersistence, MockSampler, NoopMetrics,
    Session, TurnInput, TurnOptions, default_session_path,
};

#[tokio::main]
async fn main() {
    let sampler = Arc::new(MockSampler::new());
    sampler.push_text("turn-one");
    sampler.push_text("turn-two");

    let agent = AgentBuilder::named("assistant")
        .instructions("You are helpful.")
        .model("mock")
        .build()
        .expect("build agent");

    // In-memory for demo; production: FilePersistence::for_session(root, session_id).
    let store = MemoryPersistence::default();
    let mut session = Session::new();

    let (handle, o1) = session
        .open_checkpointed_turn(
            &agent,
            sampler.as_ref(),
            &store,
            TurnInput::Text("hi".into()),
            TurnOptions::default(),
            &NoopMetrics,
        )
        .await
        .expect("turn 1");
    println!("turn1: {}", o1.output_text);
    handle.shutdown().await;

    let (handle2, o2) = session
        .open_checkpointed_turn(
            &agent,
            sampler.as_ref(),
            &store,
            TurnInput::Text("again".into()),
            TurnOptions::default(),
            &NoopMetrics,
        )
        .await
        .expect("turn 2");
    println!("turn2: {}", o2.output_text);

    let snap = handle2.snapshot().await.expect("snapshot");
    println!("messages: {}", snap.messages.len());
    println!("usage_total: {}", snap.usage.main.total_tokens);

    let loaded = store.load().await.expect("load").expect("snapshot");
    println!("checkpoint_messages: {}", loaded.messages.len());
    assert_eq!(
        loaded.messages.len(),
        snap.messages.len(),
        "checkpoint must match live handle snapshot"
    );

    let file_hint = default_session_path(std::env::temp_dir(), "sess_demo");
    let _file_store = FilePersistence::new(&file_hint);
    println!("default_path={}", file_hint.display());

    handle2.shutdown().await;
    println!("session_checkpoint_ok=true");
}
