use crate::BotContext;
use crate::BotError;
use std::sync::Arc;

#[poise::command(slash_command, track_edits)]
pub async fn mission(
    ctx: poise::Context<'_, Arc<BotContext>, BotError>,
    #[description = "Set a new system prompt and reset history"] new_mission: String,
) -> Result<(), BotError> {
    let state = ctx.data();

    {
        let mut system_prompt = state.system_prompt.lock().unwrap();
        *system_prompt = new_mission.clone();

        println!("Updated system prompt: {}", *system_prompt);
    }

    {
        let mut history = state.chat_history.lock().unwrap();
        history.clear();
    }

    ctx.say(format!(
        "The Renaissance begins anew. The new system prompt is:\n\n**{}**",
        new_mission
    ))
    .await
    .map_err(|e| {
        eprintln!("Error sending message: {}", e);
        BotError
    })?;
    Ok(())
}
