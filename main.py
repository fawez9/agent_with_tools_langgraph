from agent import interact_with_agent
from memory import get_conversation_history

config = {"configurable": {"thread_id": "2"}}

while True:
    input_message = input("You: ")
    if input_message.lower() in ["q", "quit", "exit"]:
        break
    
    result, ai_response, tool_output = interact_with_agent(input_message, config["configurable"]["thread_id"])
    
    if tool_output is not None:
        print(f"Tool output: {tool_output}")
    print(f"AI: {ai_response}")

# Display conversation history
print("\nConversation History:")
conversation_history = get_conversation_history(config["configurable"]["thread_id"])
for record in conversation_history:
    print(f"{record[3]} ({record[2]}): {record[4]}")

