class EscalationAgent:
    def __init__(self, similarity_threshold=0.45, always_escalate_priorities=("Critical", "High")):
        self.similarity_threshold = similarity_threshold
        self.always_escalate_priorities = set(always_escalate_priorities)

    def decide(self, similarity, ticket_priority=None):
        if ticket_priority in self.always_escalate_priorities:
            return {"action": "escalate",
                    "reason": f"Priority '{ticket_priority}' — routed to a human regardless of confidence."}
        if similarity < self.similarity_threshold:
            return {"action": "escalate",
                    "reason": f"Retrieval confidence too low ({similarity:.2f})."}
        return {"action": "auto_resolve",
                "reason": f"High confidence match ({similarity:.2f}), priority '{ticket_priority}' is safe."}
