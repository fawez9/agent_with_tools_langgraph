from agent import interact_with_agent
from memory import get_conversation_history, load_sessions
import uuid

def start_new_session():
    return str(uuid.uuid4())

def main():
    # Load the list of sessions from the database
    sessions = load_sessions()

    while True:
        print("\n1. Start New Session")
        print("2. Continue Existing Session")
        print("3. Quit")
        choice = input("Choose an option: ")

        if choice == "3":
            break
        elif choice == "1":
            thread_id = start_new_session()
            sessions[thread_id] = "Active"
            print(f"New session started. Session ID: {thread_id}")
        elif choice == "2":
            if not sessions:
                print("No sessions found. Please start a new session.")
                continue

            print("Sessions:")
            for idx, (session_id, status) in enumerate(sessions.items(), 1):
                print(f"{idx}. {session_id} - {status}")

            session_choice = input("Enter the number of the session to continue: ")
            try:
                session_index = int(session_choice) - 1
                thread_id = list(sessions.keys())[session_index]
            except (ValueError, IndexError):
                print("Invalid session number. Please try again.")
                continue
        else:
            print("Invalid option. Please try again.")
            continue

        print(f"Session {thread_id}")
        while True:
            input_message = input("You: ")
            if input_message.lower() in ["q", "quit", "exit"]:
                break
            
            result, ai_response, tool_output = interact_with_agent(input_message, thread_id)
            
            if tool_output is not None:
                print(f"Tool output: {tool_output}")
            print(f"AI: {ai_response}")
            
        # Display conversation history
        print("\nConversation History:")
        conversation_history = get_conversation_history(thread_id)
        for record in conversation_history:
            print(f"{record[3]} ({record[2]}): {record[4]}")

if __name__ == "__main__":
    main()