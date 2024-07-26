# agent.py
from venv import logger
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode
from tools import add, subtract
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
import os
from typing import Literal
from langgraph.checkpoint.sqlite import SqliteSaver

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

#define memory
memory = SqliteSaver.from_conn_string(":memory:")
# Initialize memory to persist state
app = workflow.compile(checkpointer=memory)

# Function to interact with the agent
def interact_with_agent(message, thread_id):
    return app.invoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": thread_id}}
    )
