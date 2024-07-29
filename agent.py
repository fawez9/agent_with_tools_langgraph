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
from memory import store_conversation

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

# Function to interact with the agent
def interact_with_agent(message, thread_id):
    result = app.invoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": thread_id}}
    )
    # Store the human message
    store_conversation(thread_id, "human", message)
    
    # Process AI responses and tool usage
    ai_response = ""
    tool_output = None
    for msg in result['messages']:
        if isinstance(msg, AIMessage):
            ai_response = msg.content.strip()
        elif isinstance(msg, ToolMessage):
            tool_output = msg.content
    
    # Store the AI response
    store_conversation(thread_id, "ai", ai_response)
    return result, ai_response, tool_output if tool_output else None