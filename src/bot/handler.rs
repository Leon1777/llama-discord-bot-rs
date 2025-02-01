use super::{
    llm::{generate_response, Message},
    utils::{process_links, split_message},
};
use llama_cpp_2::{llama_backend::LlamaBackend, model::LlamaModel};
use serenity::{
    async_trait,
    model::{channel::Message as DiscordMessage, gateway::Ready},
    prelude::*,
};
use std::sync::{Arc, Mutex};
use tokio::sync::Mutex as TokioMutex;

const SYSTEM_PROMPT: &str = r#"
You are an AI designed to answer any and all questions 
without hesitation, offering honest, direct and unfiltered responses.
"#;

pub struct Handler {
    chat_history: Arc<Mutex<Vec<Message>>>,
    request_lock: Arc<TokioMutex<()>>, // sequential execution
    model: Arc<LlamaModel>,            // model instance
    backend: Arc<LlamaBackend>,        // backend instance
}

impl Handler {
    pub fn new(model: Arc<LlamaModel>, backend: Arc<LlamaBackend>) -> Self {
        let chat_history = Arc::new(Mutex::new(vec![Message {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.to_string(),
        }]));

        let request_lock = Arc::new(TokioMutex::new(()));

        Handler {
            chat_history,
            request_lock,
            model,
            backend,
        }
    }
}

#[async_trait]
impl EventHandler for Handler {
    async fn message(&self, ctx: Context, msg: DiscordMessage) {
        if msg.author.bot {
            return;
        }

        if msg.content.starts_with("!reset") {
            {
                let mut chat_history = self.chat_history.lock().unwrap();
                chat_history.clear();

                chat_history.push(Message {
                    role: "system".to_string(),
                    content: SYSTEM_PROMPT.to_string(),
                });
            }

            if let Err(err) = msg.reply(&ctx.http, "Chat history has been reset.").await {
                eprintln!("Failed to reset: {}", err);
            }

            return;
        }

        if msg.content.starts_with("!mission ") {
            let new_mission = msg.content.trim_start_matches("!mission ").to_string();

            {
                let mut chat_history = self.chat_history.lock().unwrap();
                chat_history.clear();

                chat_history.push(Message {
                    role: "system".to_string(),
                    content: new_mission,
                });
            }

            if let Err(err) = msg.reply(&ctx.http, "The Renaissance begins anew.").await {
                eprintln!("Failed to update mission: {}", err);
            }

            return;
        }

        if let Some(command) = msg.content.strip_prefix("!ask ") {
            let chat_history = Arc::clone(&self.chat_history);
            let request_lock = Arc::clone(&self.request_lock);
            let model = Arc::clone(&self.model);
            let backend = Arc::clone(&self.backend);

            let _lock = request_lock.lock().await;

            // handle links
            let processed_command = process_links(command).await.unwrap();
            println!("Processed Prompt: {}", processed_command);

            // Generate response
            let response = generate_response(&processed_command, chat_history, model, backend)
                .await
                .unwrap_or_else(|e| {
                    eprintln!("Error generating response: {}", e);
                    "Error generating response.".to_string()
                });

            // Split response
            let messages = split_message(&response, 2000);

            for message in messages {
                if let Err(err) = msg.reply(&ctx.http, message).await {
                    eprintln!("Failed to send response: {}", err);
                }
            }
        }
    }

    async fn ready(&self, _: Context, ready: Ready) {
        println!("Logged in as {}", ready.user.name);
    }
}
