from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.backend.services.appointment_service import AppointmentService
from src.backend.services.context_manager import build_turn_context
from src.backend.services.dialogue_state_manager import (
    mark_workflow_awaiting_input,
    update_dialogue_state,
)
from src.backend.services.llm_query_analyzer import get_query_analyzer

logger = logging.getLogger(__name__)

Intent = Literal[
    "appointment",
    "navigation",
    "patient_documents",
    "hospital_knowledge",
    "general",
    "unknown",
]

WorkflowStatus = Literal[
    "in_progress",
    "awaiting_user_input",
    "completed",
    "failed",
]


class ChatState(TypedDict, total=False):
    question: str
    retrieval_question: str
    user_id: str
    conversation_id: str
    messages: Annotated[list[AnyMessage], add_messages]

    intent: Intent
    query_analysis: dict[str, Any]
    entities: dict[str, Any]
    context: dict[str, Any]
    dialogue_slots: dict[str, Any]
    dialogue_state: dict[str, Any]
    active_workflow: dict[str, Any] | None

    evidence: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    workflow_status: WorkflowStatus
    error: str | None
    answer: str


def prepare_request(state: ChatState) -> dict[str, Any]:
    """Prepare this turn without discarding persisted conversation state."""
    question = str(state.get("question") or "").strip()

    if not question:
        return {
            "question": "",
            "retrieval_question": "",
            "evidence": [],
            "tool_results": [],
            "error": "empty_question",
            "workflow_status": "failed",
        }

    return {
        "question": question,
        "retrieval_question": question,
        "entities": dict(state.get("entities") or {}),
        "dialogue_slots": dict(state.get("dialogue_slots") or {}),
        "dialogue_state": dict(state.get("dialogue_state") or {}),
        "active_workflow": state.get("active_workflow"),
        "evidence": [],
        "tool_results": [],
        "error": None,
        "workflow_status": "in_progress",
    }


def load_turn_context(state: ChatState) -> dict[str, Any]:
    """Build contextual information for the query analyzer."""
    context = build_turn_context(state)

    context["entities"] = dict(state.get("entities") or {})
    context["dialogue_slots"] = dict(state.get("dialogue_slots") or {})
    context["active_workflow"] = state.get("active_workflow")
    context["dialogue_state"] = state.get("dialogue_state") or {}

    return {"context": context}


def analyze_query(state: ChatState) -> dict[str, Any]:
    """Analyze the current turn while keeping historical state domain-scoped."""
    question = str(state.get("question") or "").strip()
    context = dict(state.get("context") or {})
    prior_entities = dict(state.get("entities") or {})
    previous_workflow = state.get("active_workflow") or {}
    previous_dialogue_state = state.get("dialogue_state") or {}

    context["entities"] = prior_entities
    context["dialogue_slots"] = dict(state.get("dialogue_slots") or {})
    context["active_workflow"] = previous_workflow
    context["dialogue_state"] = previous_dialogue_state

    analysis = get_query_analyzer().analyze(question, context)
    intent = analysis["intent"]
    current_entities = dict(analysis.get("entities") or {})

    logger.warning(
        "MEDIGUIDE DEBUG | stage=analyzer | question=%r | intent=%r | "
        "is_follow_up=%r | entities=%r",
        question,
        analysis.get("intent"),
        analysis.get("is_follow_up"),
        current_entities,
    )

    workflow_slots = (
        dict(previous_workflow.get("slots") or {})
        if isinstance(previous_workflow, dict)
        else {}
    )
    active_appointment = (
        isinstance(previous_workflow, dict)
        and previous_workflow.get("type") == "appointment"
        and previous_workflow.get("status") not in {"completed", "failed"}
    )
    pending_question = previous_dialogue_state.get("pending_question")
    current_action = current_entities.get("appointment_action")

    normalized_question = re.sub(
        r"\s+", " ", question.casefold()
    ).strip()
    normalized_action_text = re.sub(
        r"[^a-z0-9]+", " ", normalized_question
    ).strip()

    pending_cancel_slots = workflow_slots
    is_cancel_confirmation_reply = (
        pending_cancel_slots.get("pending_cancel_confirmation") is True
        and normalized_action_text in {
            "confirm", "yes", "yes cancel", "confirm cancellation",
            "proceed", "no", "no cancel", "cancel request",
            "do not cancel", "dont cancel", "stop",
        }
    )

    explicit_cancel_request = bool(
        re.search(
            r"\b(cancel|cancellation|canceling|cancelling)\b",
            normalized_question,
        )
    ) and not is_cancel_confirmation_reply

    explicit_list_request = bool(
        re.search(
            r"\b(list|show|view|display)\b.{0,35}\b(my|upcoming|all)\b.{0,20}\bappointments?\b|"
            r"\b(list|view|display)\b.{0,20}\bappointments?\b|"
            r"\b(my|upcoming|all)\s+appointments?\b|"
            r"\bwhat appointments do i have\b|\bappointments? linked to my account\b",
            normalized_question,
        )
    )

    explicit_availability_request = bool(
        re.search(
            r"\b(avail(?:able|iable) dates?|avail(?:able|iable)ity|"
            r"avail(?:able|iable) slots?|free slots?|open slots?)\b",
            normalized_question,
        )
    )

    if explicit_cancel_request:
        intent = "appointment"
        current_entities["appointment_action"] = "cancel"
        analysis = {
            **analysis,
            "intent": "appointment",
            "needs_clarification": False,
            "clarification_question": None,
            "entities": current_entities,
        }
    elif explicit_list_request:
        intent = "appointment"
        current_entities["appointment_action"] = "check"
        analysis = {
            **analysis,
            "intent": "appointment",
            "needs_clarification": False,
            "clarification_question": None,
            "entities": current_entities,
        }
    elif explicit_availability_request:
        intent = "appointment"
        current_entities["appointment_action"] = "book"
        analysis = {
            **analysis,
            "intent": "appointment",
            "needs_clarification": False,
            "clarification_question": None,
            "entities": current_entities,
        }

    current_action = current_entities.get("appointment_action")
    starts_new_booking = (
        intent == "appointment"
        and bool(
            re.search(
                r"\b(?:book|schedule|make)\b.{0,50}\bappointment\b",
                normalized_question,
            )
        )
        and not bool(
            re.search(
                r"\b(?:confirm|proceed|book it|confirm it)\b",
                normalized_question,
            )
        )
    )

    if starts_new_booking:
        selection_keys = {
            "department_or_doctor",
            "department",
            "department_name",
            "specialty",
            "department_id",
            "doctor_id",
            "doctor_name",
            "doctor",
            "display_name",
            "selected_doctor",
            "selected_department",
            "doctor_choices",
            "lookup_type",
            "destination",
            "location_code",
            "floor",
        }
        question_folded = normalized_question

        for key in selection_keys:
            value = current_entities.get(key)

            if key in {
                "department_id", "doctor_id", "selected_doctor",
                "selected_department", "doctor_choices", "lookup_type",
                "destination", "location_code", "floor",
            }:
                current_entities.pop(key, None)
            elif (
                value is not None
                and str(value).casefold() not in question_folded
            ):
                current_entities.pop(key, None)

        analysis = {**analysis, "entities": current_entities}
        current_action = current_entities.get("appointment_action")

    saved_action = (
        workflow_slots.get("appointment_action")
        or previous_workflow.get("action")
    )

    normalized_confirmation = re.sub(
        r"[^a-z0-9]+", " ", question.casefold()
    ).strip()

    pending_cancel_confirmation = (
        workflow_slots.get("pending_cancel_confirmation") is True
        and normalized_confirmation in {
            "confirm", "yes", "yes cancel", "confirm cancellation",
            "proceed", "no", "no cancel", "cancel request",
            "do not cancel", "dont cancel", "stop",
        }
    )

    if pending_cancel_confirmation:
        intent = "appointment"
        current_entities["appointment_action"] = "cancel"
        current_action = "cancel"
        analysis = {
            **analysis,
            "intent": "appointment",
            "needs_clarification": False,
            "clarification_question": None,
            "entities": current_entities,
        }

    explicit_new_operation = (
        starts_new_booking
        or (
            bool(current_action)
            and current_action != saved_action
        )
    )

    resume_appointment = (
        active_appointment
        and previous_workflow.get("status") == "awaiting_user_input"
        and bool(pending_question)
        and not explicit_new_operation
        and intent in {"appointment", "unknown", "general"}
    )

    if resume_appointment:
        intent = "appointment"

        if not current_entities.get("appointment_action"):
            saved_action = (
                workflow_slots.get("appointment_action")
                or previous_workflow.get("action")
            )
            if saved_action:
                current_entities["appointment_action"] = saved_action

        analysis = {
            **analysis,
            "intent": "appointment",
            "needs_clarification": False,
            "clarification_question": None,
            "entities": current_entities,
        }

    slots_for_manager = workflow_slots if resume_appointment else {}
    state_for_manager = dict(previous_dialogue_state)
    workflow_for_manager = previous_workflow

    if intent == "appointment" and not resume_appointment:
        suspended = state_for_manager.get("suspended_workflows") or []

        if isinstance(suspended, list):
            state_for_manager["suspended_workflows"] = [
                item for item in suspended
                if not (
                    isinstance(item, dict)
                    and item.get("type") == "appointment"
                )
            ]

        state_for_manager["active_workflow"] = None
        workflow_for_manager = None

    dialogue_state = update_dialogue_state(
        intent=intent,
        entities=current_entities,
        analysis=analysis,
        previous_dialogue_state=state_for_manager,
        active_workflow=workflow_for_manager,
        previous_dialogue_slots=slots_for_manager,
    )
    active_workflow = dialogue_state.get("active_workflow")

    entities = dict(current_entities)

    if bool(analysis.get("is_follow_up")) and intent == "navigation":
        for key in (
            "destination", "location_code", "floor", "department_name"
        ):
            if not entities.get(key) and prior_entities.get(key):
                entities[key] = prior_entities[key]

    analysis = {
        **analysis,
        "intent": intent,
        "entities": current_entities,
    }

    slots = dict(slots_for_manager)
    if isinstance(active_workflow, dict):
        slots.update(active_workflow.get("slots") or {})

    logger.warning(
        "MEDIGUIDE DEBUG | stage=dialogue_state | starts_new_booking=%r | "
        "resume_appointment=%r | current_entities=%r | active_workflow=%r | "
        "dialogue_slots=%r",
        starts_new_booking,
        resume_appointment,
        current_entities,
        active_workflow,
        slots,
    )

    return {
        "intent": intent,
        "query_analysis": analysis,
        "entities": entities,
        "dialogue_slots": slots,
        "dialogue_state": dialogue_state,
        "active_workflow": active_workflow,
        "retrieval_question": analysis.get("resolved_query") or question,
        "workflow_status": "in_progress",
        "error": None,
    }


def route_intent(state: ChatState) -> str:
    """Route by the current intent, preserving genuine workflow follow-ups."""
    analysis = state.get("query_analysis") or {}
    intent = state.get("intent", "unknown")
    workflow = state.get("active_workflow") or {}

    active_appointment = (
        isinstance(workflow, dict)
        and workflow.get("type") == "appointment"
        and workflow.get("status") not in {"completed", "failed"}
    )

    if active_appointment and (
        intent == "appointment"
        or (
            intent in {"unknown", "general"}
            and bool(analysis.get("is_follow_up"))
        )
    ):
        return "appointment"

    if analysis.get("needs_clarification"):
        return "clarification"

    if intent == "appointment":
        return "appointment"
    if intent == "hospital_knowledge":
        return "hospital_knowledge"
    if intent == "general":
        return "general"
    if intent in {"navigation", "patient_documents"}:
        return "deferred"

    return "unsupported"


def handle_clarification(state: ChatState) -> dict[str, Any]:
    """Return a clarification request from the query analyzer."""
    analysis = state.get("query_analysis") or {}
    answer = analysis.get("clarification_question")

    if not isinstance(answer, str) or not answer.strip():
        answer = "Could you clarify what you need help with?"

    dialogue_state = mark_workflow_awaiting_input(
        state.get("dialogue_state") or {},
        answer,
    )

    return {
        "answer": answer,
        "dialogue_state": dialogue_state,
        "active_workflow": dialogue_state.get("active_workflow"),
        "workflow_status": "awaiting_user_input",
        "error": None,
        "messages": [AIMessage(content=answer)],
    }

def handle_appointment(state: ChatState) -> dict[str, Any]:
    """
    Handle appointment requests through the existing database-backed service.
    """
    question = str(state.get("question") or "").strip()
    user_id = state.get("user_id")

    entities = dict(state.get("entities") or {})
    dialogue_state = dict(state.get("dialogue_state") or {})
    workflow = dict(state.get("active_workflow") or {})
    workflow_slots = dict(workflow.get("slots") or {})

    logger.warning(
        "MEDIGUIDE DEBUG | stage=appointment_handler | question=%r | "
        "entities=%r | workflow_slots=%r | workflow=%r",
        question,
        entities,
        workflow_slots,
        workflow,
    )

    action = (
        entities.get("appointment_action")
        or workflow_slots.get("appointment_action")
        or workflow.get("action")
    )

    def normalize(value: Any) -> str:
        return re.sub(
            r"[^a-z0-9]+", " ", str(value or "").casefold()
        ).strip()

    def respond(
        answer: str,
        status: WorkflowStatus = "awaiting_user_input",
        error: str | None = None,
    ) -> dict[str, Any]:
        nonlocal workflow

        if status == "completed":
            workflow = {}
            dialogue_state["active_workflow"] = None
            dialogue_state.pop("pending_question", None)
            next_slots: dict[str, Any] = {}
        else:
            workflow = {
                "type": "appointment",
                "action": action,
                "status": status,
                "slots": dict(workflow_slots),
            }
            dialogue_state["active_workflow"] = workflow
            dialogue_state["pending_question"] = answer
            next_slots = dict(workflow_slots)

        return {
            "answer": answer,
            "entities": entities,
            "dialogue_slots": next_slots,
            "active_workflow": workflow or None,
            "dialogue_state": dialogue_state,
            "workflow_status": status,
            "error": error,
            "messages": [AIMessage(content=answer)],
        }

    if not user_id:
        return respond(
            "Your authenticated identity is missing. Please sign in again.",
            "failed",
            "missing_authenticated_user",
        )

    def value(*keys: str) -> Any:
        for source in (entities, workflow_slots):
            for key in keys:
                candidate = source.get(key)
                if candidate is not None and str(candidate).strip():
                    return candidate
        return None

    def match_records(
        records: list[dict[str, Any]],
        field: str,
        utterance: str,
        id_field: str,
    ) -> list[dict[str, Any]]:
        """Match user text to real records, with conservative name matching."""
        normalized_utterance = normalize(utterance)
        utterance_tokens = set(normalized_utterance.split())
        matches: dict[str, dict[str, Any]] = {}

        for record in records:
            name = str(record.get(field) or "").strip()
            normalized_name = normalize(name)

            if not normalized_name:
                continue

            tokens = {
                token
                for token in normalized_name.split()
                if token not in {"dr", "doctor"} and len(token) > 1
            }

            exact_or_phrase = (
                normalized_name == normalized_utterance
                or bool(
                    re.search(
                        rf"(?<![a-z0-9]){re.escape(normalized_name)}(?![a-z0-9])",
                        normalized_utterance,
                    )
                )
            )

            token_match = bool(tokens) and tokens.issubset(utterance_tokens)

            if exact_or_phrase or token_match:
                record_id = str(record.get(id_field) or "")
                if record_id:
                    matches[record_id] = record

        return list(matches.values())

    try:
        service = AppointmentService()

        if action == "check":
            appointments = service.list_my_appointments(user_id)

            if not appointments:
                return respond(
                    "No appointments were returned for your linked account.",
                    "completed",
                )

            lines = [
                (
                    f"- Reference: {item.get('booking_reference')}; "
                    f"Doctor: {item.get('doctor_name')}; "
                    f"Department: {item.get('department_name')}; "
                    f"Start: {item.get('starts_at')}; "
                    f"Status: {item.get('status')}"
                )
                for item in appointments
            ]

            return respond(
                "Appointments linked to your account:\n" + "\n".join(lines),
                "completed",
            )

        if action == "cancel":
            appointments = service.list_my_appointments(
                user_id,
                include_past=False,
            )
            active_appointments = [
                item for item in appointments
                if str(item.get("status", "")).casefold() == "booked"
            ]

            normalized_text = normalize(question)
            affirmative = normalized_text in {
                "confirm", "yes", "yes cancel",
                "confirm cancellation", "proceed",
            }
            negative = normalized_text in {
                "no", "no cancel", "cancel request",
                "do not cancel", "dont cancel", "stop",
            }
            pending_cancel = (
                workflow_slots.get("pending_cancel_confirmation") is True
            )

            if pending_cancel and negative:
                workflow_slots.pop("pending_cancel_confirmation", None)
                workflow_slots.pop("cancellation_booking_reference", None)
                workflow_slots.pop("cancellation_appointment", None)
                workflow_slots.pop("cancellation_choices", None)
                return respond(
                    "Okay, I haven't cancelled the appointment.",
                    "completed",
                )

            if pending_cancel and affirmative:
                booking_reference = workflow_slots.get(
                    "cancellation_booking_reference"
                )

                if not booking_reference:
                    return respond(
                        "I couldn't find the pending appointment selection. "
                        "Please start the cancellation request again."
                    )

                selected = next(
                    (
                        item for item in active_appointments
                        if str(item.get("booking_reference", "")).casefold()
                        == str(booking_reference).casefold()
                    ),
                    None,
                )

                if selected is None:
                    workflow_slots.pop("pending_cancel_confirmation", None)
                    workflow_slots.pop("cancellation_booking_reference", None)
                    workflow_slots.pop("cancellation_appointment", None)
                    return respond(
                        "I couldn't find an active appointment with that "
                        "reference in your account. It may already be "
                        "cancelled or no longer active.",
                        "completed",
                    )

                result = service.cancel(
                    authenticated_user_id=user_id,
                    booking_reference=str(booking_reference),
                )

                workflow_slots.pop("pending_cancel_confirmation", None)
                workflow_slots.pop("cancellation_booking_reference", None)
                workflow_slots.pop("cancellation_appointment", None)
                workflow_slots.pop("cancellation_choices", None)

                return respond(
                    "Appointment cancellation completed successfully.\n"
                    f"Reference: {result.get('booking_reference')}\n"
                    f"Status: {result.get('status')}\n"
                    f"Cancelled at: {result.get('cancelled_at')}",
                    "completed",
                )

            reference_match = re.search(
                r"\bMG-[A-Z0-9]{6,}\b", question.upper()
            )
            selected = None

            if reference_match:
                requested_reference = reference_match.group(0)
                selected = next(
                    (
                        item for item in active_appointments
                        if str(item.get("booking_reference", "")).upper()
                        == requested_reference
                    ),
                    None,
                )

                if selected is None:
                    return respond(
                        "I couldn't find an active appointment with that "
                        "reference in your account. No appointment was changed.",
                        "completed",
                    )

            choices = workflow_slots.get("cancellation_choices")
            number_match = re.fullmatch(r"\s*(\d{1,2})\s*", question)

            if selected is None and number_match and isinstance(choices, list):
                index = int(number_match.group(1)) - 1

                if 0 <= index < len(choices):
                    selected_reference = choices[index].get("booking_reference")
                    selected = next(
                        (
                            item for item in active_appointments
                            if str(item.get("booking_reference", "")).casefold()
                            == str(selected_reference or "").casefold()
                        ),
                        None,
                    )

                if selected is None:
                    return respond(
                        "That selection is no longer available. Please choose "
                        "an active appointment from the list."
                    )

            if selected is None and len(active_appointments) == 1:
                selected = active_appointments[0]

            if selected is not None:
                workflow_slots["cancellation_booking_reference"] = (
                    selected["booking_reference"]
                )
                workflow_slots["cancellation_appointment"] = {
                    "booking_reference": selected["booking_reference"],
                    "doctor_name": selected.get("doctor_name"),
                    "department_name": selected.get("department_name"),
                    "starts_at": str(selected.get("starts_at")),
                }
                workflow_slots["pending_cancel_confirmation"] = True
                workflow_slots.pop("cancellation_choices", None)

                return respond(
                    "Please confirm cancellation of this appointment:\n"
                    f"Reference: {selected.get('booking_reference')}\n"
                    f"Doctor: {selected.get('doctor_name')}\n"
                    f"Department: {selected.get('department_name')}\n"
                    f"Start: {selected.get('starts_at')}\n\n"
                    "Reply 'confirm' to cancel it, or 'no' to keep it."
                )

            if not active_appointments:
                return respond(
                    "I couldn't find any active booked appointments to cancel "
                    "on your account.",
                    "completed",
                )

            workflow_slots["cancellation_choices"] = [
                {
                    "booking_reference": item.get("booking_reference"),
                    "doctor_name": item.get("doctor_name"),
                    "department_name": item.get("department_name"),
                    "starts_at": str(item.get("starts_at")),
                }
                for item in active_appointments
            ]

            lines = [
                (
                    f"{index}. {item.get('doctor_name')} — "
                    f"{item.get('department_name')} — "
                    f"{item.get('starts_at')} — "
                    f"Reference: {item.get('booking_reference')}"
                )
                for index, item in enumerate(active_appointments, start=1)
            ]

            return respond(
                "Which appointment would you like to cancel?\n"
                + "\n".join(lines)
                + "\n\nReply with the number or booking reference. "
                "I'll ask you to confirm before cancelling."
            )

        if action == "reschedule":
            return respond(
                "Chat-based rescheduling is not implemented yet. "
                "No appointment was changed. Please use the authenticated "
                "appointment API for rescheduling."
            )

        if action != "book":
            return respond(
                "I couldn't determine the appointment action. "
                "Please specify whether you want to book or check an appointment."
            )

        departments = service.list_departments()
        department = None

        saved_department_id = workflow_slots.get("department_id")
        if saved_department_id:
            department = next(
                (
                    item for item in departments
                    if str(item["department_id"]) == str(saved_department_id)
                ),
                None,
            )

        department_matches = match_records(
            departments, "name", question, "department_id"
        )

        if len(department_matches) == 1:
            department = department_matches[0]
        elif len(department_matches) > 1:
            options = "\n".join(
                f"- {item['name']}" for item in department_matches
            )
            return respond(
                "Your message matches multiple departments. "
                "Please specify one:\n" + options
            )

        if department is None:
            department_candidate = value(
                "department",
                "department_name",
                "specialty",
                "department_or_doctor",
            )

            if department_candidate:
                candidate = normalize(department_candidate)
                matches = [
                    item for item in departments
                    if normalize(item["name"]) == candidate
                ]

                if len(matches) == 1:
                    department = matches[0]
                elif len(matches) > 1:
                    return respond(
                        "Please provide the exact department name."
                    )

        doctors = service.list_doctors(
            department["department_id"] if department else None
        )

        doctor_matches = match_records(
            doctors, "display_name", question, "doctor_id"
        )

        if not doctor_matches:
            doctor_candidate = value(
                "doctor_name", "doctor", "display_name"
            )

            if doctor_candidate:
                doctor_matches = match_records(
                    doctors,
                    "display_name",
                    str(doctor_candidate),
                    "doctor_id",
                )

        doctor = None

        if len(doctor_matches) == 1:
            doctor = doctor_matches[0]
        elif len(doctor_matches) > 1:
            options = "\n".join(
                f"- {item['display_name']}" for item in doctor_matches
            )
            return respond(
                "Your message matches multiple doctors. "
                "Please specify one:\n" + options
            )

        if doctor is None:
            saved_doctor_id = workflow_slots.get("doctor_id")

            if saved_doctor_id:
                doctor = next(
                    (
                        item for item in doctors
                        if str(item["doctor_id"]) == str(saved_doctor_id)
                    ),
                    None,
                )

        if doctor is None:
            if department is None:
                global_doctors = service.list_doctors()
                global_matches = match_records(
                    global_doctors,
                    "display_name",
                    question,
                    "doctor_id",
                )

                if len(global_matches) == 1:
                    doctor = global_matches[0]
                    department = next(
                        (
                            item for item in departments
                            if str(item["department_id"])
                            == str(doctor["department_id"])
                        ),
                        None,
                    )
                    doctors = global_doctors

                elif len(global_matches) > 1:
                    options = "\n".join(
                        f"- {item['display_name']} "
                        f"({item['department_name']})"
                        for item in global_matches
                    )
                    return respond(
                        "Your message matches multiple doctors. "
                        "Please specify one:\n" + options
                    )

            if doctor is None:
                if department is None:
                    return respond(
                        "Which department or doctor would you like to see?"
                    )

                workflow_slots.update({
                    "department_id": str(department["department_id"]),
                    "department_name": department["name"],
                })

                department_doctors = service.list_doctors(
                    department["department_id"]
                )

                if not department_doctors:
                    return respond(
                        f"No active doctors were returned for "
                        f"{department['name']}."
                    )

                workflow_slots["doctor_choices"] = [
                    {
                        "doctor_id": str(item["doctor_id"]),
                        "display_name": item["display_name"],
                    }
                    for item in department_doctors
                ]

                names = "\n".join(
                    f"- {item['display_name']}"
                    for item in department_doctors
                )

                return respond(
                    f"Doctors listed under {department['name']}:\n"
                    f"{names}\n\nWhich doctor would you prefer?"
                )
        workflow_slots.update({
            "doctor_id": str(doctor["doctor_id"]),
            "doctor_name": doctor["display_name"],
            "department_id": str(doctor["department_id"]),
            "department_name": doctor["department_name"],
        })
        workflow_slots.pop("doctor_choices", None)

        raw_date = value("appointment_date", "requested_date", "date")
        requested_date = None

        if raw_date:
            try:
                requested_date = date.fromisoformat(str(raw_date)[:10])
            except (TypeError, ValueError):
                requested_date = None

        if requested_date is None:
            date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", question)
            if date_match:
                try:
                    requested_date = date.fromisoformat(date_match.group(1))
                except ValueError:
                    requested_date = None

        if requested_date is None:
            show_dates_requested = bool(
                re.search(
                    r"\b(avail(?:able|iable) dates?|show avail(?:able|iable)ity|"
                    r"check avail(?:able|iable)ity|what dates|which dates|"
                    r"when is .*available|free dates?)\b",
                    question.casefold(),
                )
            )

            if show_dates_requested:
                today = date.today()
                date_results: list[tuple[date, int]] = []

                for day_offset in range(0, 31):
                    candidate_day = today + timedelta(days=day_offset)
                    day_slots = service.get_availability(
                        doctor["doctor_id"], candidate_day
                    )
                    if day_slots:
                        date_results.append((candidate_day, len(day_slots)))
                    if len(date_results) >= 10:
                        break

                workflow_slots.pop("available_slots", None)

                if not date_results:
                    workflow_slots.pop("available_dates", None)
                    return respond(
                        f"I couldn't find available dates for "
                        f"{doctor['display_name']} in the next 30 days. "
                        "You can try another doctor or check again later."
                    )

                workflow_slots["available_dates"] = [
                    day.isoformat() for day, _count in date_results
                ]
                date_lines = [
                    f"{day.isoformat()} — {count} available slot(s)"
                    for day, count in date_results
                ]
                return respond(
                    f"Available dates for {doctor['display_name']} "
                    f"({doctor['department_name']}) in the next 30 days:\n"
                    + "\n".join(date_lines)
                    + "\n\nReply with one date in YYYY-MM-DD format "
                    "to see its available times."
                )

            return respond(
                f"You selected {doctor['display_name']} in "
                f"{doctor['department_name']}. What date would you prefer? "
                "You can say 'show available dates' or enter YYYY-MM-DD."
            )

        workflow_slots.pop("available_dates", None)
        workflow_slots["appointment_date"] = requested_date.isoformat()
        available_slots = service.get_availability(
            doctor["doctor_id"], requested_date
        )

        if not available_slots:
            workflow_slots.pop("pending_confirmation", None)
            workflow_slots.pop("starts_at", None)
            workflow_slots.pop("available_slots", None)
            return respond(
                f"No available slots were returned for "
                f"{doctor['display_name']} on {requested_date.isoformat()}. "
                "Please choose another date."
            )

        pending = workflow_slots.get("pending_confirmation") is True
        normalized_text = normalize(question)
        affirmative = normalized_text in {
            "yes", "yes confirm", "confirm", "confirm booking", "book it",
            "proceed", "proceed with booking",
        }
        negative = normalized_text in {"no", "no cancel", "do not book", "stop"}

        if pending and negative:
            workflow_slots.pop("pending_confirmation", None)
            workflow_slots.pop("starts_at", None)
            return respond("No appointment was created.")

        if pending and affirmative:
            selected_start = workflow_slots.get("starts_at")
            if not selected_start:
                workflow_slots.pop("pending_confirmation", None)
                return respond(
                    "The selected time is no longer stored in the workflow. "
                    "Please select an available slot again."
                )
            result = service.book(
                authenticated_user_id=user_id,
                doctor_id=doctor["doctor_id"],
                starts_at=selected_start,
            )
            workflow_slots.pop("pending_confirmation", None)
            workflow_slots.pop("starts_at", None)
            workflow_slots.pop("available_slots", None)
            return respond(
                "Appointment booking completed by the appointment service.\n"
                f"Reference: {result['booking_reference']}\n"
                f"Doctor: {result.get('doctor_name', doctor['display_name'])}\n"
                f"Department: {result.get('department_name', doctor['department_name'])}\n"
                f"Start: {result.get('starts_at')}\n"
                f"Status: {result.get('status')}",
                "completed",
            )

        requested_start = value("starts_at", "appointment_start")
        requested_time = value("appointment_time", "requested_time", "time")
        selected_slot = None

        if requested_start:
            selected_slot = next(
                (item for item in available_slots
                 if str(item["starts_at"]) == str(requested_start)),
                None,
            )

        if selected_slot is None and requested_time:
            for item in available_slots:
                slot_start = datetime.fromisoformat(item["starts_at"])
                if slot_start.strftime("%H:%M") == str(requested_time).strip():
                    selected_slot = item
                    break

        if selected_slot is None and workflow_slots.get("available_slots"):
            number_match = re.fullmatch(r"\s*(\d{1,2})\s*", question)
            if number_match:
                index = int(number_match.group(1)) - 1
                if 0 <= index < min(len(available_slots), 10):
                    selected_slot = available_slots[index]

        if selected_slot is None:
            workflow_slots["available_slots"] = available_slots
            lines = [
                f"{index}. "
                f"{datetime.fromisoformat(item['starts_at']).strftime('%Y-%m-%d %H:%M')}"
                for index, item in enumerate(available_slots[:10], start=1)
            ]
            timezone_name = available_slots[0].get("timezone", "local time")
            return respond(
                f"Available slots for {doctor['display_name']} on "
                f"{requested_date.isoformat()} ({timezone_name}):\n"
                + "\n".join(lines)
                + "\n\nReply with a slot number or exact HH:MM time."
            )

        workflow_slots["starts_at"] = selected_slot["starts_at"]
        workflow_slots["pending_confirmation"] = True
        return respond(
            "Please confirm this appointment before I create it:\n"
            f"Doctor: {doctor['display_name']}\n"
            f"Department: {doctor['department_name']}\n"
            f"Start: {selected_slot['starts_at']}\n"
            f"End: {selected_slot['ends_at']}\n\n"
            "Reply 'confirm' to book, or 'no' to cancel."
        )

    except Exception:
        logger.exception("Appointment workflow failed.")
        return respond(
            "I couldn't complete this appointment step. The operation "
            "was not confirmed as successful. Check the current appointment "
            "state before retrying a booking.",
            "failed",
            "appointment_workflow_failed",
        )


def answer_hospital_question(
    state: ChatState,
    rag_service: Any,
) -> dict[str, Any]:
    """Use PostgreSQL for directory facts and RAG for hospital knowledge."""
    question = (
        state.get("retrieval_question")
        or state.get("question", "")
    ).strip()

    if not question:
        answer = "Please tell me what hospital information you need."
        return {
            "answer": answer,
            "workflow_status": "awaiting_user_input",
            "messages": [AIMessage(content=answer)],
        }

    analysis = state.get("query_analysis") or {}
    current_entities = analysis.get("entities") or {}

    if not isinstance(current_entities, dict):
        current_entities = {}

    entities = dict(state.get("entities") or {})
    lookup_type = current_entities.get("lookup_type")

    def respond(
        answer: str,
        *,
        updated_entities: dict[str, Any] | None = None,
        evidence: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "answer": answer,
            "entities": (
                updated_entities if updated_entities is not None
                else entities
            ),
            "evidence": evidence or [],
            "active_workflow": state.get("active_workflow"),
            "dialogue_state": state.get("dialogue_state") or {},
            "workflow_status": "completed",
            "error": None,
            "messages": [AIMessage(content=answer)],
        }

    try:
        service = AppointmentService()

        department_name = next(
            (
                current_entities.get(key)
                for key in (
                    "department_name", "department", "specialty", "destination"
                )
                if isinstance(current_entities.get(key), str)
                and current_entities[key].strip()
            ),
            None,
        )

        doctor_name = next(
            (
                current_entities.get(key)
                for key in ("doctor_name", "doctor", "display_name")
                if isinstance(current_entities.get(key), str)
                and current_entities[key].strip()
            ),
            None,
        )

        if lookup_type == "department_location":
            if not department_name:
                return respond(
                    "Which department's location would you like to know?"
                )

            department = service.find_department(department_name)

            if department is None:
                return respond(
                    f"I couldn't verify an active department named "
                    f"'{department_name}' in the hospital directory."
                )

            answer = (
                f"{department['name']} is located at "
                f"{department['location_code']}"
            )

            if department.get("floor_label"):
                answer += f" on {department['floor_label']}"

            answer += "."

            entities.update({
                "destination": department["name"],
                "department_name": department["name"],
                "location_code": department["location_code"],
                "floor": department.get("floor_label"),
            })

            return respond(answer, updated_entities=entities)

        if lookup_type == "department_doctors":
            if not department_name:
                return respond(
                    "Which department would you like the doctor directory for?"
                )

            department = service.find_department(department_name)

            if department is None:
                return respond(
                    f"I couldn't verify an active department named "
                    f"'{department_name}' in the hospital directory."
                )

            doctors = service.list_doctors(department["department_id"])

            if not doctors:
                return respond(
                    f"No active doctors are currently listed under "
                    f"{department['name']}."
                )

            names = "\n".join(
                f"- {doctor['display_name']}" for doctor in doctors
            )

            return respond(
                f"Active doctors listed under {department['name']}:\n{names}"
            )

        if lookup_type == "doctor_department":
            if not doctor_name:
                return respond(
                    "Which doctor's department would you like me to check?"
                )

            normalized_name = re.sub(
                r"[^a-z0-9]+", " ", doctor_name.casefold()
            ).strip()

            doctors = service.list_doctors()

            matches = [
                doctor for doctor in doctors
                if re.sub(
                    r"[^a-z0-9]+", " ",
                    str(doctor.get("display_name") or "").casefold(),
                ).strip() == normalized_name
            ]

            if not matches:
                return respond(
                    f"I couldn't find an active doctor matching "
                    f"'{doctor_name}' in the hospital directory."
                )

            if len(matches) > 1:
                options = "\n".join(
                    f"- {doctor['display_name']} "
                    f"({doctor['department_name']})"
                    for doctor in matches
                )

                return respond(
                    "I found multiple matching directory records. "
                    "Please clarify which doctor you mean:\n" + options
                )

            doctor = matches[0]

            return respond(
                f"{doctor['display_name']} is listed under "
                f"{doctor['department_name']}."
            )

        result = rag_service.ask(question)

        return respond(
            result["answer"],
            evidence=result.get("evidence", []),
        )

    except Exception:
        logger.exception("Hospital knowledge retrieval failed.")

        answer = (
            "I'm unable to retrieve verified hospital information right now. "
            "Please try again shortly."
        )

        return {
            "answer": answer,
            "evidence": [],
            "workflow_status": "failed",
            "error": "hospital_knowledge_retrieval_failed",
            "messages": [AIMessage(content=answer)],
        }


def handle_general(state: ChatState) -> dict[str, Any]:
    """Handle ordinary social messages."""
    question = (
        (state.get("query_analysis") or {}).get("resolved_query")
        or state.get("question", "")
    ).strip().lower()

    if any(term in question for term in ("thank you", "thanks")):
        answer = "You're welcome. How else can I help you?"
    elif any(term in question for term in ("goodbye", "bye", "see you")):
        answer = "Goodbye. Take care."
    else:
        answer = (
            "Hello! I'm MediGuide, your hospital information assistant. "
            "I can help with hospital information and supported "
            "hospital processes."
        )

    return {
        "answer": answer,
        "dialogue_state": state.get("dialogue_state") or {},
        "workflow_status": "completed",
        "error": None,
        "messages": [AIMessage(content=answer)],
    }


def handle_deferred_intent(state: ChatState) -> dict[str, Any]:
    """Be transparent when navigation or patient-document services are absent."""
    intent = state.get("intent")

    if intent == "navigation":
        destination = (state.get("entities") or {}).get("destination")

        if not isinstance(destination, str) or not destination.strip():
            answer = (
                "Which department or hospital location would you like me to "
                "guide you to? I need a confirmed destination before I can "
                "request directions."
            )
        else:
            answer = (
                f"I understand that you need directions to {destination}. "
                "The robot navigation service is not connected yet, so I cannot "
                "generate or execute a verified route."
            )

    elif intent == "patient_documents":
        answer = (
            "The secure patient-document service is not connected yet. "
            "I cannot retrieve patient-specific documents through the "
            "general hospital knowledge system."
        )
    else:
        answer = "That service is not connected yet. No action was performed."

    return {
        "answer": answer,
        "dialogue_state": state.get("dialogue_state") or {},
        "workflow_status": "awaiting_user_input",
        "error": None,
        "messages": [AIMessage(content=answer)],
    }


def handle_unsupported(state: ChatState) -> dict[str, Any]:
    """Ask for clarification when intent is unknown."""
    answer = (
        "I'm not sure what you need help with. Could you rephrase your "
        "request or tell me whether it concerns hospital information, "
        "an appointment, or your own documents?"
    )

    return {
        "answer": answer,
        "dialogue_state": state.get("dialogue_state") or {},
        "workflow_status": "awaiting_user_input",
        "error": None,
        "messages": [AIMessage(content=answer)],
    }


def build_chat_graph(
    rag_service: Any,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Build and compile the MediGuide conversation graph."""
    graph = StateGraph(ChatState)

    graph.add_node("prepare_request", prepare_request)
    graph.add_node("load_turn_context", load_turn_context)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("clarification", handle_clarification)
    graph.add_node("appointment", handle_appointment)

    graph.add_node(
        "hospital_knowledge",
        lambda state: answer_hospital_question(state, rag_service),
    )

    graph.add_node("general", handle_general)
    graph.add_node("deferred", handle_deferred_intent)
    graph.add_node("unsupported", handle_unsupported)

    graph.add_edge(START, "prepare_request")
    graph.add_edge("prepare_request", "load_turn_context")
    graph.add_edge("load_turn_context", "analyze_query")

    graph.add_conditional_edges(
        "analyze_query",
        route_intent,
        {
            "clarification": "clarification",
            "appointment": "appointment",
            "hospital_knowledge": "hospital_knowledge",
            "general": "general",
            "deferred": "deferred",
            "unsupported": "unsupported",
        },
    )

    for node in (
        "clarification",
        "appointment",
        "hospital_knowledge",
        "general",
        "deferred",
        "unsupported",
    ):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=checkpointer)
