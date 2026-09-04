import os
from groq import Groq

class ResponseAgent:
    def __init__(self, mode="llm", model="openai/gpt-oss-20b"):
        self.mode = mode
        self.model = model
        if mode == "llm":
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                print("[ResponseAgent] WARNING: GROQ_API_KEY not set — falling back to template mode.")
                self.mode = "template"
            else:
                self.client = Groq(api_key=api_key)

    def draft_reply(self, customer_name, product, ticket_description, kb_answer):
        if self.mode == "template":
            return (f"Hi {customer_name},\n\nThanks for reaching out about your {product}. "
                    f"{kb_answer}\n\nLet us know if this resolves the issue or if you need "
                    f"further help.\n\nBest,\nSupport Team")
        prompt = (f"You are a customer support agent. Write a short, warm, professional reply.\n\n"
                  f"Customer name: {customer_name}\nProduct: {product}\n"
                  f"Customer's issue: {ticket_description}\nReference solution: {kb_answer}\n\n"
                  f"Write only the reply text, 3-5 sentences.")
        response = self.client.chat.completions.create(
            model=self.model, max_tokens=300,
            messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
