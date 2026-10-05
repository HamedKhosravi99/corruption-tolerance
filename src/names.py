"""Display names for Chatbot Arena model identifiers, shared by the figures, the tables and
the text, so that each model has one name in the paper. Table 5 (all twenty pairs) keeps
the raw identifiers, which carry the release dates."""

ARENA = {
    "gpt-4o-2024-05-13": "GPT-4o",
    "gpt-4-turbo-2024-04-09": "GPT-4-Turbo",
    "gpt-4-1106-preview": "GPT-4-1106",
    "gpt-4-0125-preview": "GPT-4-0125",
    "claude-3-5-sonnet-20240620": "Claude-3.5-Sonnet",
    "claude-3-opus-20240229": "Claude-3-Opus",
    "claude-3-sonnet-20240229": "Claude-3-Sonnet",
    "claude-3-haiku-20240307": "Claude-3-Haiku",
    "gemini-1.5-pro-api-0409-preview": "Gemini-1.5-Pro-0409",
    "gemini-1.5-pro-api-0514": "Gemini-1.5-Pro-0514",
    "gemini-advanced-0514": "Gemini-Advanced-0514",
    "yi-large-preview": "Yi-Large-Preview",
    "llama-3-70b-instruct": "Llama-3-70B",
    "llama-3-8b-instruct": "Llama-3-8B",
    "snowflake-arctic-instruct": "Snowflake-Arctic",
}


def arena(model_id):
    """Display name of an Arena model identifier (the identifier itself if it is not listed)."""
    return ARENA.get(model_id, model_id)
