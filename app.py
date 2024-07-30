import streamlit as st
from main import SessionManager

# Create an instance of the SessionManager class
session_manager = SessionManager()

# Streamlit app
st.set_page_config(page_title="AI Agent Interaction", page_icon="🤖")
st.title("AI Agent Interaction")

# Initialize session state
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to handle sending messages
def send_message():
    user_message = st.session_state.user_input
    st.session_state.user_input = ""  # Clear the input before processing

    if user_message:
        try:
            result, ai_response, tool_output = session_manager.agent.interact_with_agent(user_message, st.session_state.thread_id)
            st.session_state.messages.append({"role": "human", "content": user_message})
            st.session_state.messages.append({"role": "ai", "content": ai_response})
            if tool_output is not None:
                st.session_state.messages.append({"role": "tool", "content": tool_output})
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Sidebar for session management
st.sidebar.title("Session Management")
option = st.sidebar.selectbox(
    "Choose an option:",
    ["Start New Session", "Continue Existing Session"]
)

if option == "Start New Session":
    if st.sidebar.button("Start New Session"):
        st.session_state.messages = []
        st.session_state.thread_id = session_manager.start_new_session()
        st.sidebar.success(f"New session started. Session ID: {st.session_state.thread_id}")

elif option == "Continue Existing Session":
    sessions = session_manager.sessions
    if not sessions:
        st.sidebar.warning("No sessions found. Please start a new session.")
    else:
        session_id = st.sidebar.selectbox(
            "Select a session to continue:",
            list(sessions.keys())
        )
        if st.sidebar.button("Load Session"):
            st.session_state.thread_id = session_id
            st.session_state.messages = []
            conversation_history = session_manager.db_handler.get_conversation_history(st.session_state.thread_id)
            for record in conversation_history:
                role = record[3]
                content = record[4]
                st.session_state.messages.append({"role": role, "content": content})
            st.sidebar.success(f"Continuing session: {st.session_state.thread_id}")

# Main chat interface
if st.session_state.thread_id:
    st.write(f"Session {st.session_state.thread_id}")

    # Chat input
    if "messages" not in st.session_state.keys():
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Display chat messages and bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = session_manager.agent.interact_with_agent(prompt, st.session_state.thread_id)
                full_response = response['output_text'][0]
                st.write(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.info("Please start a new session or continue an existing one.")
