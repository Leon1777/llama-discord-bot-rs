mod bot;

use bot::handler::Handler;
use dotenv::dotenv;
use llama_cpp_2::{
    llama_backend::LlamaBackend,
    model::{params::LlamaModelParams, LlamaModel},
};
use serenity::model::gateway::GatewayIntents;
use serenity::Client;
use std::env;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    dotenv().ok();
    let token = env::var("DISCORD_TOKEN").expect("Expected DISCORD_TOKEN in environment variables");
    let intents = GatewayIntents::GUILD_MESSAGES | GatewayIntents::MESSAGE_CONTENT;

    // Initialize backend and model
    let backend = Arc::new(LlamaBackend::init().expect("Failed to initialize Llama backend"));

    let model_path = "models/Mistral-Nemo-Instruct-2407-abliterated_q8_0.gguf";
    let model_params = LlamaModelParams::default()
        .with_n_gpu_layers(41)
        .with_use_mlock(true);

    let model = Arc::new(
        LlamaModel::load_from_file(&backend, model_path, &model_params)
            .expect("Failed to load Llama model"),
    );

    let handler = Handler::new(model, backend);

    let mut client = Client::builder(token, intents)
        .event_handler(handler)
        .await
        .expect("Error creating client");

    if let Err(why) = client.start().await {
        println!("Client error: {:?}", why);
    }
}
