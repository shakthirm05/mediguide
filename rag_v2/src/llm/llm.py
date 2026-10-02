import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen3-1.7B"


class QwenLLM:

    def __init__(self):

        self.model_name = MODEL_NAME

        print("\nLoading LLM tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        print("Tokenizer loaded.")

        print("\nLoading LLM model...")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )

        self.model.eval()

        print("\nLLM loaded successfully.")
        print(f"Device: {self.model.device}")

    # ======================================================
    # LOW-LEVEL GENERATION
    # ======================================================

    def _generate_tokens(
        self,
        messages,
        max_new_tokens=200
    ):

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False
        )

        input_device = (
            self.model
            .get_input_embeddings()
            .weight
            .device
        )

        inputs = {
            key: value.to(input_device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[-1]:
        ]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        return response

    # ======================================================
    # RESPONSE CLEANING
    # ======================================================

    def _clean_response(
        self,
        response: str
    ):

        response = response.strip()

        # Remove completed thinking blocks
        if "<think>" in response:

            if "</think>" in response:

                response = response.split(
                    "</think>",
                    1
                )[1].strip()

            else:

                response = response.replace(
                    "<think>",
                    ""
                ).strip()

        return response

    # ======================================================
    # RAG ANSWER GENERATION
    # ======================================================

    def generate(
        self,
        question: str,
        context: str
    ):

        system_prompt = """
You are the hospital information assistant for
MediGuide General Hospital.

Your job is to answer the patient's question using
ONLY the hospital information provided in the context.

Rules:

1. Give ONLY the final answer to the patient.

2. Do NOT describe your reasoning process.

3. Do NOT mention evidence, context, retrieval,
documents, prompts, or the language model.

4. Do NOT say things such as:
   "Let me check",
   "Let me see",
   "The user is asking",
   "According to the evidence",
   or "Looking at the context".

5. Do NOT use your pretrained knowledge to invent
hospital-specific information.

6. Do NOT invent hospital policies.

7. Do NOT invent hospital procedures.

8. Do NOT invent doctors.

9. Do NOT invent departments.

10. Do NOT invent services or facilities.

11. Do NOT invent hospital locations.

12. If the provided context does not contain enough
information to answer the question, say:

"I don't have enough information in the hospital
knowledge base to answer that."

13. Keep the answer concise and easy for a patient
to understand.

14. Do not provide medical diagnosis.

15. Do not prescribe medication.

16. Do not provide treatment recommendations.

17. For emergencies, provide only operational
information available in the supplied hospital
context.
"""

        user_prompt = f"""
Hospital Information:

{context}

Patient Question:

{question}

Provide only the final patient-facing answer.
"""

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        print("\nGenerating response...")

        response = self._generate_tokens(
            messages,
            max_new_tokens=200
        )

        response = self._clean_response(
            response
        )

        if not response:

            response = (
                "I don't have enough information in "
                "the hospital knowledge base to "
                "answer that."
            )

        return response

    # ======================================================
    # RAW GENERATION
    # Used by Query Understanding
    # ======================================================

    def generate_raw(
        self,
        prompt: str,
        max_new_tokens=250
    ):

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = self._generate_tokens(
            messages,
            max_new_tokens=max_new_tokens
        )

        response = self._clean_response(
            response
        )

        return response