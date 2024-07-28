# agent.py
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode
from tools import add, subtract
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
import os
from typing import Literal
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from datetime import datetime
import json

# Load environment variables
load_dotenv()

# Configure the Google GenAI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

tools = [add, subtract]

# Define the tool node
tool_node = ToolNode(tools)

# Bind the tools to the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# Define the function that calls the model
def call_model(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    # We return a list, because this will get added to the existing list
    return {"messages": response}

def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    """Return the next node to execute."""
    messages = state['messages']
    if messages[-1].tool_calls:
        return "tools"
    return "__end__"

# Define the workflow
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", 'agent')

# Define memory with a persistent SQLite database file
memory = SqliteSaver.from_conn_string("agent_memory.db")

# Initialize memory to persist state
app = workflow.compile(checkpointer=memory)

# Function to store conversation in memory
def store_conversation(thread_id, role, content):
    with sqlite3.connect("agent_memory.db") as conn:
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO conversations (thread_id, timestamp, role, content)
            VALUES (?, ?, ?, ?)
        """, (thread_id, timestamp, role, content))
        conn.commit()

# Function to store tool usage in memory
def store_tool_usage(thread_id, tool_name, input_data, output_data):
    with sqlite3.connect("agent_memory.db") as conn:
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO tool_usage (thread_id, timestamp, tool_name, input_data, output_data)
            VALUES (?, ?, ?, ?, ?)
        """, (thread_id, timestamp, tool_name, json.dumps(input_data), json.dumps(output_data)))
        conn.commit()

# Function to interact with the agent
def interact_with_agent(message, thread_id):
    result = app.invoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": thread_id}}
    )
    
    # Store the human message
    store_conversation(thread_id, "human", message)
    
    # Process and store AI responses and tool usage
    for msg in result['messages']:
        if isinstance(msg, AIMessage):
            store_conversation(thread_id, "ai", msg.content)
        elif isinstance(msg, ToolMessage):
            store_tool_usage(thread_id, msg.tool_call_id, message, msg.content)
    
    return result

# Function to get conversation history
def get_conversation_history(thread_id):
    with sqlite3.connect("agent_memory.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conversations WHERE thread_id = ? ORDER BY timestamp", (thread_id,))
        return cursor.fetchall()

# Function to get tool usage history
def get_tool_usage_history(thread_id):
    with sqlite3.connect("agent_memory.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tool_usage WHERE thread_id = ? ORDER BY timestamp", (thread_id,))
        return cursor.fetchall()

# Initialize the database tables
def initialize_db():
    with sqlite3.connect("agent_memory.db") as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT,
                timestamp TEXT,
                role TEXT,
                content TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tool_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT,
                timestamp TEXT,
                tool_name TEXT,
                input_data TEXT,
                output_data TEXT
            )
        """)
        conn.commit()

# Main execution block
if __name__ == "__main__":
    print("Initializing database...")
    initialize_db()
    print("Database initialized successfully.")