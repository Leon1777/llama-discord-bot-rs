pub fn split_message(content: &str, max_length: usize) -> Vec<String> {
    let mut result = Vec::new();
    let mut remaining = content;

    while !remaining.is_empty() {
        if remaining.len() <= max_length {
            result.push(remaining.to_string());
            break;
        }

        // find last whitespace within max_length
        let split_at = match remaining[..max_length].rfind(char::is_whitespace) {
            Some(pos) => pos,
            None => max_length, // split at max_length
        };

        let (part, rest) = remaining.split_at(split_at);

        result.push(part.trim_end().to_string());
        remaining = rest.trim_start();
    }

    result
}
