from dialogue_state import DialogueState


state = DialogueState(
    session_id="session_001"
)


# Turn 1
print("\nUSER:")
print("I have an appointment.")

state.set_intent("appointment")
state.set_goal(
    "help patient with appointment"
)

state.add_missing_information("doctor")
state.set_awaiting("doctor_name")

state.display()


# Turn 2
print("\nUSER:")
print("Dr. Arun.")

state.update_entity(
    "doctor",
    "Dr. Arun"
)

state.remove_missing_information("doctor")
state.clear_awaiting()

state.display()


# Turn 3
print("\nUSER:")
print("Where should I go?")

state.set_intent(
    "appointment_navigation"
)

state.set_goal(
    "help patient reach appointment department"
)

state.update_entity(
    "destination",
    "department associated with Dr. Arun"
)

state.display()