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

    messages = [
        {
            "role": "system",
            "content": """
            Classify the user request into ONE word only:
            DATA -> database query
            REPORT -> generate report
            EMAIL -> send email
            GENERAL -> greeting or normal conversation

            Respond with only one word.
            """
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    try:
        response = ask_llm(messages)
        return response.strip().upper()
    except:
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

    # If no memory provided
    return ask_llm([{"role": "user", "content": user_query}])