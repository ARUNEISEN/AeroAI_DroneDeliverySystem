from agents.local_llm import ask_llm
from agents.Data_Agent import handle_data_query
from agents.Report_Agent import handle_report
from agents.Email_Agent import handle_email


def route_query(user_query, memory=None):

    decision = classify_intent(user_query)

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

    prompt = f"""
    Classify the user request:
    DATA -> database query
    REPORT -> generate report
    EMAIL -> send email
    GENERAL -> greeting or normal conversation

    Only return one word.

    Query: {user_query}
    """

    try:
        response = ask_llm(prompt)
        return response.strip().upper()
    except:
        return "GENERAL"


# ---------------------------------------
# General Conversation
# ---------------------------------------

def handle_general(user_query, memory):

    if memory:
        memory.add("user", user_query)
        response = ask_llm(memory.get_context())
        memory.add("assistant", response)
        return response

    return ask_llm(user_query)
