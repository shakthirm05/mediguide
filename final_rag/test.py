
from src.backend.services.dialogue_state_manager import update_dialogue_state

# Turn 1: begin an appointment workflow.
appointment = update_dialogue_state(
    intent="appointment",
    entities={"appointment_action": "book"},
    analysis={
        "intent": "appointment",
        "resolved_query": "Book an appointment",
        "is_follow_up": False,
        "needs_clarification": False,
    },
)

# Simulate an appointment workflow that is waiting for user input.
active_appointment = appointment["active_workflow"]
active_appointment["status"] = "awaiting_user_input"
active_appointment["slots"]["appointment_action"] = "book"

# Turn 2: user interrupts with a navigation request.
navigation = update_dialogue_state(
    intent="navigation",
    entities={
        "destination": "Cardiology",
        "location_code": "C-201",
        "floor": "Level 2",
    },
    analysis={
        "intent": "navigation",
        "resolved_query": "Give directions to Cardiology",
        "is_follow_up": True,
        "needs_clarification": False,
    },
    previous_dialogue_state=appointment,
    active_workflow=active_appointment,
    previous_dialogue_slots=active_appointment["slots"],
)

assert navigation["active_workflow"] is None
assert len(navigation["suspended_workflows"]) == 1
assert navigation["suspended_workflows"][0]["type"] == "appointment"
assert navigation["entities_by_domain"]["navigation"]["destination"] == "Cardiology"

# Turn 3: user returns to appointments.
resumed = update_dialogue_state(
    intent="appointment",
    entities={},
    analysis={
        "intent": "appointment",
        "resolved_query": "Continue the appointment request",
        "is_follow_up": True,
        "needs_clarification": False,
    },
    previous_dialogue_state=navigation,
    previous_dialogue_slots={},
)

assert resumed["active_workflow"]["type"] == "appointment"
assert resumed["active_workflow"]["slots"]["appointment_action"] == "book"
assert resumed["suspended_workflows"] == []

print("PASS: appointment workflow can be suspended and resumed.")
print("Suspended during navigation:", navigation["suspended_workflows"])
print("Resumed workflow:", resumed["active_workflow"])
