from agents.local_llm import ask_llm
from agents.Data_Agent import handle_data_query
from agents.Report_Agent import handle_report
from agents.Email_Agent import handle_email
import logging
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def route_query(user_query, memory=None):

    logging.info(f"Incoming Query: {user_query}")

    decision = classify_intent(user_query)

    logging.info(f"Routed To: {decision}")

    if decision == "DATA":
        return handle_data_query(user_query)

    elif decision == "REPORT":
        return handle_report(user_query)

    elif decision == "EMAIL":
        return handle_email(user_query)

    else:
        return handle_general(user_query, memory)
    
# ---------------------------------------
# LLM Intent Classification
# ---------------------------------------
def classify_intent(user_query):

    query = user_query.lower()

    # EMAIL
    if re.search(r"\S+@\S+\.\S+", query) or "send email" in query:
        return "EMAIL"

    # REPORT
    if "report" in query:
        return "REPORT"

    # DATA
    data_keywords = {"health", "status", "damage", "missing", "count", "drone"}
    if any(word in query for word in data_keywords):
        return "DATA"

    return "GENERAL"
    
# ---------------------------------------
# General Conversation
# ---------------------------------------

def handle_general(user_query, memory):

    if memory is not None:

        # Add user message
        memory.append({"role": "user", "content": user_query})

        # Send full conversation to LLM
        response = ask_llm(memory)

        # Add assistant reply
        memory.append({"role": "assistant", "content": response})

        return response

    return ask_llm([{"role": "user", "content": user_query}])