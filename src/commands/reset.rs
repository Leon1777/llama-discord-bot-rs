use crate::bot::default_prompt::SYSTEM_PROMPT;
use crate::bot::inference::Message as LlmMessage;
use crate::BotContext;
use crate::BotError;
use std::sync::Arc;

#[poise::command(slash_command, track_edits)]
pub async fn reset(ctx: poise::Context<'_, Arc<BotContext>, BotError>) -> Result<(), BotError> {
    let state = ctx.data();

    {
        let mut chat_history = state.chat_history.lock().unwrap();
        chat_history.clear();
        chat_history.push(LlmMessage {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.to_string(),
        });
    }

    ctx.say("Chat history has been reset.").await.map_err(|e| {
        eprintln!("Error sending message: {}", e);
        BotError
    })?;
    Ok(())
}
