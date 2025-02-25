from enum import StrEnum


class APIModel(StrEnum):
    DEFAULT = "gemini/gemini-2.0-flash-001"
    # DEFAULT = "claude-3-5-sonnet-20240620"
    OPENAI = "gpt-4o-2024-08-06"  # Default OpenAI model
    GPT_O3_MINI = "o3-mini-2025-01-31"
    GPT_4O_MINI = "gpt-4o-mini-2024-07-18"
    GPT_4O = "gpt-4o-2024-08-06"
    GEMINI = "gemini/gemini-2.0-flash-exp"  # Default Gemini model
    GEMINI_FLASH = "gemini/gemini-1.5-flash-002"
    GEMINI_PRO = "gemini/gemini-1.5-pro-002"
    GEMINI_FLASH_2 = "gemini/gemini-2.0-flash-001"
    GEMINI_FLASH_2_LITE = "gemini/gemini-2.0-flash-lite-preview-02-05"
    ANTHROPIC = "claude-3-5-sonnet-20240620"  # Default Anthropic model
    CLAUDE_SONNET = "claude-3-5-sonnet-20240620"
    CLAUDE_OPUS = "claude-3-opus-20240229"
    CLAUDE_HAIKU = "claude-3-5-haiku-20241022"


API_PROVIDERS = {
    APIModel.OPENAI: "OPENAI",
    APIModel.GPT_4O_MINI: "OPENAI",
    APIModel.GPT_4O: "OPENAI",
    APIModel.GEMINI: "GEMINI",
    APIModel.GEMINI_FLASH: "GEMINI",
    APIModel.GEMINI_PRO: "GEMINI",
    APIModel.GEMINI_FLASH_2: "GEMINI",
    APIModel.ANTHROPIC: "ANTHROPIC",
    APIModel.CLAUDE_SONNET: "ANTHROPIC",
    APIModel.CLAUDE_OPUS: "ANTHROPIC",
    APIModel.CLAUDE_HAIKU: "ANTHROPIC",
}
