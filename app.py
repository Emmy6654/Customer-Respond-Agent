import os
from flask import Flask, request, jsonify, send_from_directory
from agents.intent_agent import IntentAgent
from agents.retrieval_agent import RetrievalAgent
from agents.response_agent import ResponseAgent
from agents.escalation_agent import EscalationAgent

app = Flask(__name__, static_folder="static")

print("Initializing agents...")
intent_agent = IntentAgent()
intent_agent.train("data/tickets.csv")

retrieval_agent = RetrievalAgent("data/knowledge_base.json")
response_agent = ResponseAgent(mode="llm")
escalation_agent = EscalationAgent(similarity_threshold=0.35)
print("All agents ready.")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/process", methods=["POST"])
def process():
    data = request.get_json()
    customer_name = data.get("customer_name", "Customer")
    product = data.get("product", "your product")
    ticket_description = data.get("ticket_description", "")
    ticket_priority = data.get("ticket_priority", "Medium")

    if not ticket_description.strip():
        return jsonify({"error": "Ticket description is required."}), 400

    match = retrieval_agent.retrieve(ticket_description, top_k=1)[0]
    predicted_intent = intent_agent.predict(ticket_description)
    draft_reply = response_agent.draft_reply(
        customer_name, product, ticket_description, match["answer"]
    )
    escalation = escalation_agent.decide(match["similarity"], ticket_priority)

    return jsonify({
        "matched_subject": match["subject"],
        "retrieval_similarity": round(match["similarity"], 3),
        "predicted_intent": predicted_intent,
        "draft_reply": draft_reply,
        "escalation_action": escalation["action"],
        "escalation_reason": escalation["reason"],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
