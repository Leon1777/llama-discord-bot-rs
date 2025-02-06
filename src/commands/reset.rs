use crate::bot::default_prompt::SYSTEM_PROMPT;
use crate::BotContext;
use crate::BotError;
use std::sync::Arc;

#[poise::command(slash_command, track_edits)]
pub async fn reset(ctx: poise::Context<'_, Arc<BotContext>, BotError>) -> Result<(), BotError> {
    let state = ctx.data();

    {
        let mut system_prompt = state.system_prompt.lock().unwrap();
        *system_prompt = SYSTEM_PROMPT.to_string();
    }

    {
        let mut chat_history = state.chat_history.lock().unwrap();
        chat_history.clear();
    }

    ctx.say("Chat history has been reset.").await.map_err(|e| {
        eprintln!("Error sending message: {}", e);
        BotError
    })?;
    Ok(())
}
