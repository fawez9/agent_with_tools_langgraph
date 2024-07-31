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
from memory import DatabaseHandler

class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        
        # Load environment variables
        load_dotenv()
        
        # Configure the Google GenAI
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        self.tools = [add, subtract]
        
        # Define the tool node
        self.tool_node = ToolNode(self.tools)
        
        # Bind the tools to the LLM
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Define memory with a persistent SQLite database file
        self.memory = SqliteSaver.from_conn_string("agent_memory.db")
        
        # Define the workflow
        self.workflow = StateGraph(MessagesState)
        self.workflow.add_node("agent", self.call_model)
        self.workflow.add_node("tools", self.tool_node)
        self.workflow.add_edge(START, "agent")
        self.workflow.add_conditional_edges("agent", self.should_continue)
        self.workflow.add_edge("tools", 'agent')
        
        # Initialize memory to persist state
        self.app = self.workflow.compile(checkpointer=self.memory)
        
        # Initialize DatabaseHandler
        self.db_handler = DatabaseHandler()

    def call_model(self, state: MessagesState):
        response = self.llm_with_tools.invoke(state["messages"])
        # We return a list, because this will get added to the existing list
        return {"messages": response}

    def should_continue(self, state: MessagesState) -> Literal["tools", "__end__"]:
        """Return the next node to execute."""
        messages = state['messages']
        if messages[-1].tool_calls:
            return "tools"
        return "__end__"

    def interact_with_agent(self, message, thread_id):
        result = self.app.invoke(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        # Store the human message
        self.db_handler.store_conversation(thread_id, "human", message)

        # Process AI responses and tool usage
        ai_response = ""
        tool_output = None
        for msg in result['messages']:
            if isinstance(msg, AIMessage):
                ai_response = msg.content.strip()
            elif isinstance(msg, ToolMessage):
                tool_output = msg.content

        # Store the AI response
        self.db_handler.store_conversation(thread_id, "ai", ai_response)
        return {"output_text": [ai_response]}
    
    # Terminal version of the agent for testing: CAN BE REMOVED
    def interact_with_agent_terminal(self,message, thread_id):
        result = self.app.invoke(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        # Store the human message
        self.db_handler.store_conversation(thread_id, "human", message)
        
        # Process AI responses and tool usage
        ai_response = ""
        tool_output = None
        for msg in result['messages']:
            if isinstance(msg, AIMessage):
                ai_response = msg.content.strip()
            elif isinstance(msg, ToolMessage):
                tool_output = msg.content
        
        # Store the AI response
        self.db_handler.store_conversation(thread_id, "ai", ai_response)
        return result, ai_response, tool_output if tool_output else None