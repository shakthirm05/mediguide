class ContextQueryBuilder:

    def build(
        self,
        user_message: str,
        dialogue_state: dict
    ):

        intent = dialogue_state.get(
            "current_intent"
        )

        entities = dialogue_state.get(
            "entities",
            {}
        )

        doctor = entities.get("doctor")
        department = entities.get("department")
        service = entities.get("service")
        destination = entities.get("destination")

        navigation_context = dialogue_state.get(
            "navigation_context",
            {}
        )

        current_location = navigation_context.get(
            "current_location"
        )

        # --------------------------------------------------
        # DOCTOR INFORMATION
        # --------------------------------------------------

        if intent == "doctor_information":

            parts = [
                "Hospital doctor information:"
            ]

            if doctor:
                parts.append(
                    f"Doctor: {doctor}"
                )

            return " ".join(parts)

        # --------------------------------------------------
        # DOCTOR → DEPARTMENT
        # --------------------------------------------------

        if intent == "doctor_department":

            parts = [
                "Which department does the doctor "
                "belong to?"
            ]

            if doctor:
                parts.append(
                    f"Doctor: {doctor}"
                )

            return " ".join(parts)

        # --------------------------------------------------
        # DOCTOR → LOCATION / NAVIGATION
        # --------------------------------------------------

        if intent == "doctor_navigation":

            parts = [
                "Where is the doctor's department?"
            ]

            if doctor:
                parts.append(
                    f"Doctor: {doctor}"
                )

            if current_location:
                parts.append(
                    f"Starting location: "
                    f"{current_location}"
                )

            return " ".join(parts)

        # --------------------------------------------------
        # APPOINTMENT NAVIGATION
        # --------------------------------------------------

        if intent == "appointment_navigation":

            parts = [
                "Where should I go for my appointment?"
            ]

            if doctor:
                parts.append(
                    f"Doctor: {doctor}"
                )

            if department:
                parts.append(
                    f"Department: {department}"
                )

            if current_location:
                parts.append(
                    f"Starting location: "
                    f"{current_location}"
                )

            return " ".join(parts)

        # --------------------------------------------------
        # DEPARTMENT NAVIGATION
        # --------------------------------------------------

        if intent == "department_navigation":

            parts = [
                "Where is the department?"
            ]

            if department:
                parts.append(
                    f"Department: {department}"
                )

            if current_location:
                parts.append(
                    f"Starting location: "
                    f"{current_location}"
                )

            return " ".join(parts)

        # --------------------------------------------------
        # SERVICE INFORMATION
        # --------------------------------------------------

        if intent == "service_information":

            parts = [
                "Hospital service information:"
            ]

            if service:
                parts.append(
                    f"Service: {service}"
                )

            return " ".join(parts)

        # --------------------------------------------------
        # FACILITY INFORMATION
        # --------------------------------------------------

        if intent == "facility_information":

            parts = [
                "Hospital facility information:"
            ]

            if destination:
                parts.append(
                    f"Facility: {destination}"
                )

            return " ".join(parts)

        # --------------------------------------------------
        # REGISTRATION
        # --------------------------------------------------

        if intent == "registration":

            return (
                "Hospital registration information: "
                + user_message
            )

        # --------------------------------------------------
        # VISITOR INFORMATION
        # --------------------------------------------------

        if intent == "visitor_information":

            return (
                "Hospital visitor information: "
                + user_message
            )

        # --------------------------------------------------
        # EMERGENCY INFORMATION
        # --------------------------------------------------

        if intent == "emergency_information":

            return (
                "Hospital emergency information: "
                + user_message
            )

        # --------------------------------------------------
        # HOSPITAL INFORMATION
        # --------------------------------------------------

        if intent == "hospital_information":

            return (
                "Hospital information: "
                + user_message
            )

        # --------------------------------------------------
        # APPOINTMENT
        # --------------------------------------------------

        if intent == "appointment":

            parts = [
                "Hospital appointment information:"
            ]

            if doctor:
                parts.append(
                    f"Doctor: {doctor}"
                )

            if department:
                parts.append(
                    f"Department: {department}"
                )

            return " ".join(parts)

        # --------------------------------------------------
        # APPOINTMENT CANCELLATION
        # --------------------------------------------------

        if intent == "appointment_cancellation":

            return (
                "Hospital appointment cancellation "
                "information: "
                + user_message
            )

        # --------------------------------------------------
        # APPOINTMENT RESCHEDULING
        # --------------------------------------------------

        if intent == "appointment_rescheduling":

            return (
                "Hospital appointment rescheduling "
                "information: "
                + user_message
            )

        # --------------------------------------------------
        # GREETING
        # --------------------------------------------------

        if intent == "greeting":

            return user_message

        # --------------------------------------------------
        # UNKNOWN / FALLBACK
        # --------------------------------------------------

        return user_message