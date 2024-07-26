from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent import app

config = {"configurable": {"thread_id": "2"}}

def print_message(message):
    if isinstance(message, HumanMessage):
        print(f"HumanMessage content: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"AIMessage content: {message.content}")
    elif isinstance(message, ToolMessage):
        print(f"ToolMessage content: {message.content}")
    else:
        print(f"Unexpected type: {type(message).__name__} - {message}")

seen_message_ids = set()  # Track message IDs to avoid duplicates

while True:
    input_message = HumanMessage(content=input(">> "))
    if input_message.content == "q":
        break
    
    for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
        for k, v in event.items():
            if k != "__end__":
                if isinstance(v, list):
                    for item in v:
                        # Use a unique identifier for the message to track seen messages
                        item_id = getattr(item, 'id', str(item))
                        if item_id not in seen_message_ids:
                            print_message(item)
                            seen_message_ids.add(item_id)
                else:
                    # Handle non-list item
                    item_id = getattr(v, 'id', str(v))
                    if item_id not in seen_message_ids:
                        print_message(v)
                        seen_message_ids.add(item_id)
