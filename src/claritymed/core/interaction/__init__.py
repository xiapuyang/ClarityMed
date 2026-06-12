"""User-interaction primitives — ask the user a structured question mid-turn.

Sibling to ``core.rag``: declares the LLM-facing tool, the per-turn
runtime channel that hands the question off to a UI, and the Pydantic
schemas that constrain how the LLM may phrase questions.

The channel is a Protocol so the host (Textual TUI, future web, eval
runners) plugs in its own transport. ``HeadlessPromptChannel`` is the
safe default for non-interactive contexts (one-shot CLI ``ask``, evals,
tests) — it tells the LLM the channel is unavailable instead of blocking.
"""

from claritymed.core.interaction.prompt_channel import (
    HeadlessPromptChannel,
    InteractiveChannelUnavailable,
    PromptChannel,
    UserDeclinedAnswer,
)
from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    AskUserQuestionResult,
    Question,
    QuestionOption,
)
from claritymed.core.interaction.tool_approval_channel import (
    ApprovalDecision,
    HeadlessToolApprovalChannel,
    ToolApprovalChannel,
)
from claritymed.core.interaction.tools.ask_user_question import (
    ASK_USER_QUESTION_TOOL_NAME,
    build_ask_user_question_tool,
)

__all__ = [
    "ASK_USER_QUESTION_TOOL_NAME",
    "ApprovalDecision",
    "AskUserQuestionInput",
    "AskUserQuestionResult",
    "HeadlessPromptChannel",
    "HeadlessToolApprovalChannel",
    "InteractiveChannelUnavailable",
    "PromptChannel",
    "Question",
    "QuestionOption",
    "ToolApprovalChannel",
    "UserDeclinedAnswer",
    "build_ask_user_question_tool",
]
