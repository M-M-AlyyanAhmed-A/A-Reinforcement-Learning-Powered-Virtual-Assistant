import json
import os

def save_conversation(user_input, assistant_response):
    """Saves the conversation to memory."""
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONVERSATION_MEMORY = os.path.join(BASE_PATH, "conversation_memory.json")
    with open(CONVERSATION_MEMORY, 'r') as f:
        memory = json.load(f)
    memory.append({"user": user_input, "assistant": assistant_response})
    with open(CONVERSATION_MEMORY, 'w') as f:
        json.dump(memory, f, indent=4)