from agent import Agent
from memory import DatabaseHandler
import uuid

class SessionManager:
    def __init__(self):
        self.db_handler = DatabaseHandler()
        self.sessions = self.db_handler.load_sessions()

    def start_new_session(self):
        thread_id = str(uuid.uuid4())
        self.sessions[thread_id] = "Active"
        print(f"New session started. Session ID: {thread_id}")
        return thread_id

    def continue_existing_session(self):
        if not self.sessions:
            print("No sessions found. Please start a new session.")
            return None
        
        print("Sessions:")
        for idx, (session_id, status) in enumerate(self.sessions.items(), 1):
            print(f"{idx}. {session_id} - {status}")

        session_choice = input("Enter the number of the session to continue: ")
        try:
            session_index = int(session_choice) - 1
            thread_id = list(self.sessions.keys())[session_index]
            return thread_id
        except (ValueError, IndexError):
            print("Invalid session number. Please try again.")
            return None

    def interact_with_session(self, thread_id):
        print(f"Session {thread_id}")
        while True:
            input_message = input("You: ")
            if input_message.lower() in ["q", "quit", "exit"]:
                break
            
            result, ai_response, tool_output = self.agent.interact_with_agent(input_message, thread_id) # u can the interact_with_agent_terminal() for terminal mode

            if tool_output is not None:
                print(f"Tool output: {tool_output}")
            print(f"AI: {ai_response}")
        
        # Display conversation history
        print("\nConversation History:")
        conversation_history = self.db_handler.get_conversation_history(thread_id)
        for record in conversation_history:
            print(f"{record[3]} ({record[2]}): {record[4]}")
    def main(self):
        while True:
            print("\n1. Start New Session")
            print("2. Continue Existing Session")
            print("3. Quit")
            choice = input("Choose an option: ")

            if choice == "3":
                break
            elif choice == "1":
                thread_id = self.start_new_session()
            elif choice == "2":
                thread_id = self.continue_existing_session()
                if not thread_id:
                    continue
            else:
                print("Invalid option. Please try again.")
                continue

            self.interact_with_session(thread_id)

if __name__ == "__main__":
    session_manager = SessionManager()
    session_manager.main()

