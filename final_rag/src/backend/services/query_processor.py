
from __future__ import annotations

import re
from typing import Any, Literal, TypedDict


Intent = Literal[
    "appointment",
    "navigation",
    "patient_documents",
    "hospital_knowledge",
    "general",
    "unknown",
]


class QueryAnalysis(TypedDict):
    original_question: str
    normalized_question: str
    resolved_question: str
    intent: Intent
    entities: dict[str, Any]
    is_follow_up: bool
    context_resolved: bool


APPOINTMENT_TERMS = (
    "appointment",
    "appointments",
    "book a slot",
    "book an appointment",
    "schedule an appointment",
    "reschedule",
    "cancel my booking",
    "cancel appointment",
    "cancel my appointment",
    "check my appointment",
    "appointment status",
    "available slots",
)

PATIENT_DOCUMENT_TERMS = (
    "my report",
    "my reports",
    "my prescription",
    "my prescriptions",
    "my medical records",
    "my uploaded document",
    "my uploaded documents",
    "my bill",
    "my bills",
    "my scan",
    "my scans",
)

# These are explicit robot/navigation actions. They remain separate from
# ordinary questions asking for a department's location.
PHYSICAL_NAVIGATION_TERMS = (
    "guide me to",
    "take me to",
    "navigate to",
    "show me the way",
    "start navigation",
    "start navigating",
)

LOCATION_QUESTION_TERMS = (
    "where is",
    "where are",
    "where can i find",
    "location of",
    "directions to",
)

GENERAL_TERMS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "thank you",
    "thanks",
    "bye",
    "goodbye",
}

FOLLOW_UP_PATTERNS = (
    r"how do i get there",
    r"how can i get there",
    r"how do i reach there",
    r"how can i reach there",
    r"how do i go there",
    r"how can i go there",
    r"where is it",
    r"where can i find it",
    r"what about that",
    r"what about it",
    r"tell me more",
    r"and then",
    r"what next",
    r"what about there",
    r"how do i reach it",
    r"how do i get to it",
)

APPOINTMENT_ACTIONS = {
    "book": (
        "book",
        "schedule",
        "make an appointment",
    ),
    "reschedule": (
        "reschedule",
        "change my appointment",
    ),
    "cancel": (
        "cancel",
        "cancellation",
    ),
    "check": (
        "check my appointment",
        "appointment status",
    ),
}

HOSPITAL_TERMS = (
    "hospital",
    "department",
    "doctor",
    "registration",
    "admission",
    "discharge",
    "visitor",
    "visiting hours",
    "billing",
    "insurance",
    "policy",
    "procedure",
    "cardiology",
    "radiology",
    "emergency",
    "pharmacy",
    "laboratory",
    "lab",
    "floor",
    "level",
    "room",
    "help desk",
    "helpdesk",
    "cafeteria",
    "elevator",
    "lift",
)


def normalize_question(question: str) -> str:
    """Normalize casing and whitespace without changing the original."""
    return re.sub(r"\s+", " ", question).strip().lower()


def contains_term(question: str, terms: tuple[str, ...]) -> bool:
    """Match complete phrases rather than accidental substrings."""
    for term in terms:
        pattern = r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)"
        if re.search(pattern, question):
            return True
    return False


def is_follow_up(question: str) -> bool:
    """Detect common short references to a previous conversational turn."""
    normalized = normalize_question(question).rstrip("?!., ")

    return any(
        re.fullmatch(pattern, normalized)
        for pattern in FOLLOW_UP_PATTERNS
    )


def classify_intent(question: str) -> Intent:
    """Classify a question into one of MediGuide's supported intent labels."""
    question = normalize_question(question)

    if not question:
        return "unknown"

    if contains_term(question, APPOINTMENT_TERMS):
        return "appointment"

    if contains_term(question, PATIENT_DOCUMENT_TERMS):
        return "patient_documents"

    # A question asking where a department is can be answered by hospital
    # knowledge. It is not, by itself, a robot movement command.
    if contains_term(question, LOCATION_QUESTION_TERMS):
        return "hospital_knowledge"

    if contains_term(question, PHYSICAL_NAVIGATION_TERMS):
        return "navigation"

    if question in GENERAL_TERMS:
        return "general"

    if contains_term(question, HOSPITAL_TERMS):
        return "hospital_knowledge"

    return "unknown"


def extract_entities(question: str) -> dict[str, Any]:
    """
    Extract explicit mentions only.

    Extracted names are unverified mentions, not database-confirmed entities.
    """
    question = normalize_question(question)
    entities: dict[str, Any] = {}

    for action, terms in APPOINTMENT_ACTIONS.items():
        if contains_term(question, terms):
            entities["appointment_action"] = action
            break

    department_patterns = (
        r"\bwhere\s+is\s+(?:the\s+)?"
        r"([a-z][a-z0-9 -]{0,59}?)(?=\?|$|[,.])",
        r"\bwhere\s+are\s+(?:the\s+)?"
        r"([a-z][a-z0-9 -]{0,59}?)(?=\?|$|[,.])",
        r"\bwhere\s+can\s+i\s+find\s+(?:the\s+)?"
        r"([a-z][a-z0-9 -]{0,59}?)(?=\?|$|[,.])",
        r"\bdirections\s+to\s+(?:the\s+)?"
        r"([a-z][a-z0-9 -]{0,59}?)(?=\?|$|[,.])",
        r"\bnavigate\s+to\s+(?:the\s+)?"
        r"([a-z][a-z0-9 -]{0,59}?)(?=\?|$|[,.])",
        r"\bguide\s+me\s+to\s+(?:the\s+)?"
        r"([a-z][a-z0-9 -]{0,59}?)(?=\?|$|[,.])",
        r"\btake\s+me\s+to\s+(?:the\s+)?"
        r"([a-z][a-z0-9 -]{0,59}?)(?=\?|$|[,.])",
        r"\blocation\s+of\s+(?:the\s+)?"
        r"([a-z][a-z0-9 -]{0,59}?)(?=\?|$|[,.])",
    )

    for pattern in department_patterns:
        match = re.search(pattern, question)
        if match:
            candidate = match.group(1).strip(" .,!?:;")
            if candidate:
                entities["destination_mention"] = candidate
                break

    doctor_match = re.search(
        r"\bdr\.?\s+"
        r"([a-z][a-z'-]*(?:\s+[a-z][a-z'-]*){0,3})",
        question,
        flags=re.IGNORECASE,
    )

    if doctor_match:
        entities["doctor_mention"] = doctor_match.group(1).strip()

    return entities


def _message_content(message: Any) -> str:
    """Extract text from a context-manager dictionary or LangChain message."""
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = [
            item["text"]
            for item in content
            if isinstance(item, dict)
            and isinstance(item.get("text"), str)
        ]
        return "\n".join(parts).strip()

    return ""


def _message_role(message: Any) -> str | None:
    if isinstance(message, dict):
        role = message.get("role")
        return role if role in ("user", "assistant") else None

    message_type = getattr(message, "type", None)

    if message_type == "human":
        return "user"
    if message_type == "ai":
        return "assistant"

    return None


def _previous_turn(
    history: list[Any],
    current_question: str,
) -> tuple[str | None, str | None]:
    """Get the previous user question and assistant answer."""
    messages: list[tuple[str, str]] = []

    for message in history:
        role = _message_role(message)
        content = _message_content(message)

        if role and content:
            messages.append((role, content))

    # The context manager normally includes the current user message.
    # Exclude it so it cannot be mistaken for the previous question.
    if messages and messages[-1][0] == "user":
        if normalize_question(messages[-1][1]) == normalize_question(
            current_question
        ):
            messages.pop()

    previous_user = None
    previous_assistant = None

    for role, content in reversed(messages):
        if previous_assistant is None and role == "assistant":
            previous_assistant = content
        elif previous_user is None and role == "user":
            previous_user = content

        if previous_user and previous_assistant:
            break

    return previous_user, previous_assistant


def _extract_previous_topic(question: str) -> str | None:
    """Extract a subject from a simple prior location question."""
    patterns = (
        r"(?i)^where\s+is\s+(?:the\s+)?(.+)$",
        r"(?i)^where\s+are\s+(?:the\s+)?(.+)$",
        r"(?i)^where\s+can\s+i\s+find\s+(?:the\s+)?(.+)$",
        r"(?i)^location\s+of\s+(?:the\s+)?(.+)$",
    )

    cleaned = question.strip().rstrip("?!., ")

    for pattern in patterns:
        match = re.fullmatch(pattern, cleaned)
        if match:
            topic = match.group(1).strip().rstrip("?!., ")
            if topic:
                return topic

    return None


def _is_location_follow_up(question: str) -> bool:
    normalized = normalize_question(question).rstrip("?!., ")

    return normalized in {
        "how do i get there",
        "how can i get there",
        "how do i reach there",
        "how can i reach there",
        "how do i go there",
        "how can i go there",
        "how do i reach it",
        "how do i get to it",
    }


def resolve_follow_up(
    question: str,
    history: list[Any] | None = None,
) -> tuple[str, bool]:
    """
    Resolve a follow-up with previous conversational evidence.

    If there is insufficient history, keep the original question unchanged.
    """
    original = question.strip()

    if not is_follow_up(original):
        return original, False

    previous_user, previous_assistant = _previous_turn(
        history or [],
        original,
    )

    if not previous_user or not previous_assistant:
        return original, False

    topic = _extract_previous_topic(previous_user)

    if topic and _is_location_follow_up(original):
        return (
            f"Where is {topic}?\n"
            f"Previous hospital answer: {previous_assistant}\n"
            "Answer using verified hospital knowledge. Do not invent "
            "physical robot directions or claim a route is confirmed.",
            True,
        )

    return (
        f"Current follow-up: {original}\n"
        f"Previous user question: {previous_user}\n"
        f"Previous assistant answer: {previous_assistant}\n"
        "Use the conversation and verified evidence to resolve the "
        "follow-up. If its meaning remains ambiguous, ask for clarification.",
        True,
    )


def process_query(
    question: str,
    context: dict[str, Any] | None = None,
) -> QueryAnalysis:
    """Analyze a query and resolve supported follow-ups when context exists."""
    original_question = question.strip()
    normalized = normalize_question(original_question)

    context = context or {}
    history = context.get("recent_history") or []

    resolved_question, context_resolved = resolve_follow_up(
        original_question,
        history,
    )

    resolved_normalized = normalize_question(resolved_question)

    # An identified location follow-up is a hospital knowledge request,
    # not a robot navigation command.
    if context_resolved and _is_location_follow_up(original_question):
        intent: Intent = "hospital_knowledge"
    else:
        intent = classify_intent(resolved_normalized)

    entities = extract_entities(resolved_normalized)

    # Preserve previously collected entities when the current query does
    # not explicitly supply a replacement. These are still unverified
    # mentions until checked against the relevant database or service.
    prior_entities = context.get("entities") or {}

    for key, value in prior_entities.items():
        entities.setdefault(key, value)

    return {
        "original_question": original_question,
        "normalized_question": normalized,
        "resolved_question": resolved_question,
        "intent": intent,
        "entities": entities,
        "is_follow_up": is_follow_up(original_question),
        "context_resolved": context_resolved,
    }
