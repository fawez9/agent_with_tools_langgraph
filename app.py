import streamlit as st
from main import SessionManager
from agent import Agent 
from chat_Unstructured import EmbeddingManager, VectorStoreManager

# Create instances of necessary classes
session_manager = SessionManager()
agent = Agent(agent_id="streamlit_agent")

# Streamlit app
st.set_page_config(page_title="AI Agent Interaction", page_icon="🤖")
st.title("AI Agent Interaction")

# Initialize session state
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_store_folder' not in st.session_state:
    st.session_state.vector_store_folder = None

# Sidebar for session management
st.sidebar.title("Session Management")
option = st.sidebar.selectbox(
    "Choose an option:",
    ["Start New Session", "Continue Existing Session"]
)

# Handle session options
if option == "Start New Session":
    if st.sidebar.button("Start New Session"):
        st.session_state.messages = []
        st.session_state.thread_id = session_manager.start_new_session()
        st.sidebar.success(f"New session started. Session ID: {st.session_state.thread_id}")

if option == "Continue Existing Session":
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

# Add file uploader and URL input fields
with st.sidebar.form(key="file_upload_form"):
    uploaded_files = st.file_uploader(
        "Upload your Files",
        accept_multiple_files=True,
        type=['pdf', 'csv', 'txt', 'xls', 'json'],
        key="file_uploader"
    )
    url = st.text_input("Enter a URL to process", key="url_input")
    
    if st.form_submit_button("Submit & Process"):
        if uploaded_files or url:
            with st.spinner("Processing files and URL..."):
                # Save uploaded files temporarily
                file_paths = []
                for file in uploaded_files:
                    with open(file.name, "wb") as f:
                        f.write(file.getbuffer())
                    file_paths.append(file.name)
                
                # Process files and URL
                text_chunks = EmbeddingManager().process_files_and_url(file_paths, url)
                st.session_state.vector_store_folder = VectorStoreManager().create_vector_store(text_chunks)
                st.success("Files and URL processed successfully!")
        else:
            st.warning("Please upload files or enter a URL to process.")

# Main chat interface
if st.session_state.thread_id:
    # Chat input
    if "messages" not in st.session_state.keys():
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)


        # Display chat messages and bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.vector_store_folder:
                    # Use the vector store for question answering
                    response = agent.interact_with_agent(prompt,st.session_state.thread_id,st.session_state.vector_store_folder)
                    full_response = response['output_text'][0]
                    st.write(full_response)
                else:
                    # Use the regular agent interaction
                    response = agent.interact_with_agent(prompt, st.session_state.thread_id,None)
                    full_response = response['output_text'][0]
                    st.write(full_response) 
        st.session_state.messages.append({"role": "assistant", "content": full_response})


else:
    st.info("Please start a new session or continue an existing one.")