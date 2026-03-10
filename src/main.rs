/// Main entry point for the juan bridge application.
///
/// This module orchestrates the entire application by:
/// 1. Parsing CLI arguments
/// 2. Loading configuration
/// 3. Initializing managers (agent, session)
/// 4. Connecting to Slack
/// 5. Processing events in the main loop
mod agent;
mod bridge;
mod cli;
mod config;
mod handler;
mod log_timer;
mod session;
mod slack;
mod utils;

use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use tracing::info;

use crate::bridge::run_bridge;
use crate::log_timer::Iso8601Timer;

#[tokio::main]
async fn main() -> Result<()> {
    let args = cli::Args::parse();

    match args.command {
        cli::Command::Init { config, r#override } => {
            return config::Config::init(&config, r#override);
        }
        cli::Command::Run { config, log_level } => {
            // Initialize logging with ISO 8601 local timestamps
            tracing_subscriber::fmt()
                .with_env_filter(log_level)
                .with_timer(Iso8601Timer)
                .init();

            info!("Loading configuration from: {}", config);
            let config = Arc::new(config::Config::load(&config)?);

            info!("Configuration loaded successfully");
            run_bridge(config).await?;
        }
    }

    Ok(())
}
