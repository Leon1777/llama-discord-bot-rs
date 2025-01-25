use std::env;

pub fn get_discord_token() -> String {
    env::var("DISCORD_TOKEN").expect("Expected DISCORD_TOKEN in environment variables")
}
