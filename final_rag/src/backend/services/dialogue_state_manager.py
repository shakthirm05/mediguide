from __future__ import annotations

from copy import deepcopy
from typing import Any

# Only appointment currently has a persistent workflow. Navigation and patient
# documents are recognized domains, but their workflows remain deferred.
WORKFLOW_INTENTS = {"appointment": "appointment"}
ENTITY_DOMAINS = {
    "appointment",
    "navigation",
    "patient_documents",
    "hospital_knowledge",
    "general",
    "unknown",
}
APPOINTMENT_REQUIRED_SLOTS = (
    "department_or_doctor",
    "appointment_date",
    "appointment_time",
)


def _clean_entities(entities: dict[str, Any] | None) -> dict[str, Any]:
    """Remove empty values without inventing missing information."""
    if not isinstance(entities, dict):
        return {}
    return {
        key: value
        for key, value in entities.items()
        if isinstance(key, str)
        and key.strip()
        and value is not None
        and not (isinstance(value, str) and not value.strip())
    }


def _workflow_type_for_intent(intent: str) -> str | None:
    """Return a persistent workflow type for the current intent."""
    return WORKFLOW_INTENTS.get(intent)


def _missing_appointment_slots(slots: dict[str, Any]) -> list[str]:
    """Return required appointment fields that are still missing."""
    return [
        slot for slot in APPOINTMENT_REQUIRED_SLOTS
        if slots.get(slot) is None or slots.get(slot) == ""
    ]


def _find_suspended_workflow(
    suspended_workflows: list[dict[str, Any]], workflow_type: str
) -> int | None:
    """Find the most recently suspended workflow of the requested type."""
    for index in range(len(suspended_workflows) - 1, -1, -1):
        if suspended_workflows[index].get("type") == workflow_type:
            return index
    return None


def _suspend_workflow(
    workflow: dict[str, Any], suspended_workflows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Store an interrupted workflow once, preserving its structured slots."""
    suspended = deepcopy(workflow)
    if not isinstance(suspended.get("type"), str):
        return suspended_workflows
    suspended["status"] = "suspended"
    # Replace an older suspended workflow of the same type to avoid stale
    # duplicate workflows competing to resume.
    suspended_workflows[:] = [
        item for item in suspended_workflows
        if item.get("type") != suspended.get("type")
    ]
    suspended_workflows.append(suspended)
    return suspended_workflows


def update_dialogue_state(
    *,
    intent: str,
    entities: dict[str, Any] | None,
    analysis: dict[str, Any],
    previous_dialogue_state: dict[str, Any] | None = None,
    active_workflow: dict[str, Any] | None = None,
    previous_dialogue_slots: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance structured dialogue state for one turn; performs no actions.

    The caller decides whether an old workflow is eligible to resume. Passing
    ``active_workflow=None`` means no active workflow; callers that intentionally
    reset state should also clear ``previous_dialogue_state['active_workflow']``.
    """
    state = deepcopy(previous_dialogue_state or {})
    entities_by_domain = dict(state.get("entities_by_domain") or {})
    suspended_workflows = [
        deepcopy(item)
        for item in (state.get("suspended_workflows") or [])
        if isinstance(item, dict) and isinstance(item.get("type"), str)
    ]
    current_entities = _clean_entities(entities)

    domain = intent if intent in ENTITY_DOMAINS else "general"
    domain_entities = dict(entities_by_domain.get(domain) or {})
    domain_entities.update(current_entities)
    entities_by_domain[domain] = domain_entities

    # Use the explicit workflow from the graph when supplied. Otherwise retain
    # the state manager's persisted active workflow for normal follow-up turns.
    if active_workflow is None:
        active_workflow = state.get("active_workflow")
    current_workflow = (
        deepcopy(active_workflow) if isinstance(active_workflow, dict) else None
    )

    previous_slots = dict(previous_dialogue_slots or {})
    if current_workflow is not None:
        workflow_slots = dict(current_workflow.get("slots") or {})
        workflow_slots.update({
            key: value for key, value in previous_slots.items()
            if value is not None and value != ""
        })
        current_workflow["slots"] = workflow_slots

    requested_workflow_type = _workflow_type_for_intent(intent)

    # A different intent interrupts the current workflow. Entity domains stay
    # separate, so hospital-information entities don't overwrite appointment slots.
    if (
        current_workflow is not None
        and current_workflow.get("type") != requested_workflow_type
    ):
        suspended_workflows = _suspend_workflow(current_workflow, suspended_workflows)
        current_workflow = None

    # Resume a suspended workflow only when the caller routes the turn to that
    # workflow type. The graph must clear suspended appointment state for a new
    # explicit booking so this fallback cannot resurrect stale department data.
    if requested_workflow_type is not None and (
        current_workflow is None
        or current_workflow.get("type") != requested_workflow_type
    ):
        index = _find_suspended_workflow(suspended_workflows, requested_workflow_type)
        if index is not None:
            current_workflow = suspended_workflows.pop(index)
            current_workflow["status"] = "in_progress"

    if requested_workflow_type is not None and current_workflow is None:
        current_workflow = {
            "type": requested_workflow_type,
            "status": "in_progress",
            "slots": {},
        }

    if current_workflow is not None:
        workflow_slots = dict(current_workflow.get("slots") or {})
        if current_workflow.get("type") == requested_workflow_type:
            workflow_slots.update(current_entities)
        current_workflow["slots"] = workflow_slots
        if current_workflow.get("type") == "appointment":
            current_workflow["missing_slots"] = _missing_appointment_slots(workflow_slots)

    needs_clarification = bool(analysis.get("needs_clarification"))
    clarification_question = (
        analysis.get("clarification_question") if needs_clarification else None
    )
    if not isinstance(clarification_question, str):
        clarification_question = None

    state.update({
        "last_intent": intent,
        "entities_by_domain": entities_by_domain,
        "active_workflow": current_workflow,
        "suspended_workflows": suspended_workflows,
        "pending_question": clarification_question,
        "last_resolved_query": analysis.get("resolved_query"),
        "last_follow_up": bool(analysis.get("is_follow_up")),
    })
    return state


def mark_workflow_awaiting_input(
    dialogue_state: dict[str, Any], question: str
) -> dict[str, Any]:
    """Mark the current interaction as waiting for the user's response."""
    state = deepcopy(dialogue_state or {})
    state["pending_question"] = question
    workflow = state.get("active_workflow")
    if isinstance(workflow, dict):
        workflow["status"] = "awaiting_user_input"
    return state


def clear_pending_question(dialogue_state: dict[str, Any]) -> dict[str, Any]:
    """Clear a pending question when the conversation advances."""
    state = deepcopy(dialogue_state or {})
    state["pending_question"] = None
    return state
