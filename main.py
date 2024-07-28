# main.py
import os
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent import app, get_conversation_history, get_tool_usage_history, interact_with_agent

config = {"configurable": {"thread_id": "2"}}

def print_message(message):
    if isinstance(message, HumanMessage):
        print(f"Human: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"AI: {message.content}")
    elif isinstance(message, ToolMessage):
        tool_name = message.tool_call_id if hasattr(message, 'tool_call_id') else "Unknown Tool"
        print(f"Tool ({tool_name}): {message.content}")
    else:
        print(f"Unexpected type: {type(message).__name__} - {message}")

while True:
    input_message = input("You: ")
    if input_message.lower() in ["q", "quit", "exit"]:
        break
    
    result = interact_with_agent(input_message, config["configurable"]["thread_id"])
    for message in result['messages']:
        print_message(message)

# Display conversation history
print("\nConversation History:")
conversation_history = get_conversation_history(config["configurable"]["thread_id"])
for record in conversation_history:
    print(f"{record[3]} ({record[2]}): {record[4]}")

# Display tool usage history
print("\nTool Usage History:")
tool_usage_history = get_tool_usage_history(config["configurable"]["thread_id"])
for record in tool_usage_history:
    print(f"Tool: {record[3]} ({record[2]})")
    print(f"Input: {record[4]}")
    print(f"Output: {record[5]}")
    print()