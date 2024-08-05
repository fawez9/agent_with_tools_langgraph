import streamlit as st
from main import SessionManager
from agent import Agent 
from chat_Unstructured import EmbeddingManager, VectorStoreManager
import uuid
import os

# Create instances of necessary classes
session_manager = SessionManager()
# Streamlit app
st.set_page_config(page_title="AI Agent Interaction", page_icon="🤖")
st.title("AI Agent Interaction")

# Initialize session state
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent_id' not in st.session_state:
    st.session_state.agent_id = None
if 'files' not in st.session_state:
    st.session_state.files = {}  # {file_id: {'name': file_name, 'vector_store': vector_store_path}}

# Ensure session consistency
if st.session_state.thread_id and st.session_state.thread_id in session_manager.sessions:
    if not isinstance(session_manager.sessions[st.session_state.thread_id], dict):
        session_manager.sessions[st.session_state.thread_id] = {
            'agent_id': st.session_state.agent_id,
            'files': st.session_state.files
        }
elif st.session_state.thread_id:
    session_manager.sessions[st.session_state.thread_id] = {
        'agent_id': st.session_state.agent_id,
        'files': st.session_state.files
    }

# Sidebar for session management
st.sidebar.title("Session Management")
option = st.sidebar.selectbox(
    "Choose an option:",
    ["Start New Session", "Continue Existing Session"]
)


# Sidebar for session managemen
if option == "Start New Session":
    if st.sidebar.button("Start New Session"):
        st.session_state.thread_id = session_manager.start_new_session()
        st.session_state.agent_id = str(uuid.uuid4())
        st.session_state.agent = Agent(agent_id=st.session_state.agent_id)
        st.session_state.files = {}
        st.session_state.messages = []
        session_manager.sessions[st.session_state.thread_id] = {
            'agent_id': st.session_state.agent_id,  
            'files': {}
        }
        st.sidebar.success(f"New session started. Session ID: {st.session_state.thread_id}")
        # Clear existing messages
        
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
            session_data = sessions.get(session_id, {})
            if isinstance(session_data, dict):
                st.session_state.agent_id = session_data.get('agent_id')
                if not st.session_state.agent_id:
                    st.session_state.agent_id = str(uuid.uuid4())
                    session_data['agent_id'] = st.session_state.agent_id
                st.session_state.agent = Agent(agent_id=st.session_state.agent_id)
                st.session_state.files = session_data.get('files', {})
            else:
                st.session_state.agent_id = str(uuid.uuid4())
                st.session_state.agent = Agent(agent_id=st.session_state.agent_id)
                st.session_state.files = {}
                sessions[session_id] = {
                    'agent_id': st.session_state.agent_id,
                    'files': {}
                }
            
            # Update the session data to ensure it's stored correctly
            session_manager.sessions[session_id] = {
                'agent_id': st.session_state.agent_id,
                'files': st.session_state.files
            }

            conversation_history = session_manager.db_handler.get_conversation_history(st.session_state.thread_id)
            for record in conversation_history:
                role = record[3]
                content = record[4]
                st.session_state.messages.append({"role": role, "content": content})
            st.sidebar.success(f"Continuing session: {st.session_state.thread_id}")
            
            # Display loaded files
            if st.session_state.files:
                st.sidebar.write("Loaded Files:")
                for file_id, file_info in st.session_state.files.items():
                    st.sidebar.write(f"- {file_info['name']}")


# Add file uploader and URL input fields
if st.session_state.thread_id:
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
                    for uploaded_file in uploaded_files:
                        file_id = str(uuid.uuid4())
                        file_name = uploaded_file.name
                        file_path = f"uploads/{st.session_state.agent_id}/{file_id}_{file_name}"
                        
                        # Ensure the directory exists
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        
                        # Save the file
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process the file
                        text_chunks = EmbeddingManager().process_files_and_url([file_path], None)
                        vector_store_path = VectorStoreManager().create_vector_store(text_chunks, file_id)
                        
                        # Store file information
                        st.session_state.files[file_id] = {
                            'name': file_name,
                            'vector_store': vector_store_path
                        }
                    
                    # Process URL if provided
                    if url:
                        url_id = str(uuid.uuid4())
                        text_chunks = EmbeddingManager().process_files_and_url([], url)
                        vector_store_path = VectorStoreManager().create_vector_store(text_chunks, url_id)
                        st.session_state.files[url_id] = {
                            'name': url,
                            'vector_store': vector_store_path
                        }
                    
                    # Update session manager
                    session_manager.sessions[st.session_state.thread_id] = {
                        'agent_id': st.session_state.agent_id,
                        'files': st.session_state.files
                    }
                    
                    st.success("Files and URL processed successfully!")
            else:
                st.warning("Please upload files or enter a URL to process.")

    # Display all files (both loaded and newly uploaded)
    if st.session_state.files:
        st.sidebar.write("All Files:")
        for file_id, file_info in st.session_state.files.items():
            st.sidebar.write(f"- {file_info['name']}")

# Main chat interface
if st.session_state.thread_id:
    st.title("Chat Interface")
    st.write(f"Session ID: {st.session_state.thread_id} - Agent ID: {st.session_state.agent_id}")

    # Ensure agent is initialized
    if 'agent' not in st.session_state:
        st.session_state.agent = Agent(agent_id=st.session_state.agent_id)

    # Display chat messages
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
                if st.session_state.files:
                    # Use all vector stores for question answering
                    vector_stores = [file_info['vector_store'] for file_info in st.session_state.files.values()]
                    response = st.session_state.agent.interact_with_agent(prompt, st.session_state.thread_id, vector_stores)
                else:
                    # Use the regular agent interaction
                    response = st.session_state.agent.interact_with_agent(prompt, st.session_state.thread_id, None)
                full_response = response['output_text'][0]
                st.write(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.info("Please start a new session to begin.")
