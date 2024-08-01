from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode
from tools import add, subtract
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
import google.generativeai as genai
from dotenv import load_dotenv
import os
from typing import Literal
from langgraph.checkpoint.sqlite import SqliteSaver
from memory import DatabaseHandler
import pandas as pd
import io
import uuid
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import json
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from chat_Unstructured import VectorStoreManager, EmbeddingManager


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
        
        # Initialize VectorStoreManager and EmbeddingManager
        self.vector_store_manager = VectorStoreManager()
        self.embedding_manager = EmbeddingManager()

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

    def interact_with_agent(self, message, thread_id,vector_store_folder ):
        # Process file data if provided
        if vector_store_folder:
            new_db = self.vector_store_manager.load_vector_store(vector_store_folder)
            docs = new_db.similarity_search(message)
            chain = self.get_conversational_chain()
        
            # Pass the documents directly to the chain
            response = chain({"input_documents": docs, "question": message})
            # Combine the message with file content if any
            combined_message = f"""consider this as a prompt u should follow the following rules:
            this is the message_user :{message}\n
            there are the documents:{docs} \n
            and the answer of the chain is:{response['output_text']}\n
            please focus on what's the message is sometimes the user asks about the file content and sometimes he only wanna chat about anything else the provided answer can be sometimes has some wrong answer or some missing information try always to check both the docs and the answer and provide the best answer for the user u can sometimes forget about the file content or the cain answer and u do that work\n
            here's an example of the wrong answer m talking about : this can be the chain answer RTE BIZERTE "C NASR,MNIHLA 2094 OUESLATI LYES B MED " while the user asked about the client name of a bill the answer of teh chain was both a a place combined by a client name of the bill this is wrong so always check both the docs and the answer and take the chain answer as a basic answer and not really right ur work is to fix the chain answer if it is wrong or sometimes generate one by urself\n
            dont mention in ur answer anything about the prompt i gave u to follow u should answer the message_user only 
            """
        else:
            combined_message = message
        
        result = self.app.invoke(
            {"messages": [HumanMessage(content=combined_message)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        # Store the human message
        self.db_handler.store_conversation(thread_id, "human", message)

        # Process AI responses and tool usage
        ai_response = ""
        #tool_output = None
        for msg in result['messages']:
            if isinstance(msg, AIMessage):
                ai_response = msg.content.strip()
            """ elif isinstance(msg, ToolMessage):
                tool_output = msg.content """

        # Store the AI response
        self.db_handler.store_conversation(thread_id, "ai", ai_response)
        return {"output_text": [ai_response]}


    def get_conversational_chain(self):
        prompt_template = """
    You are an expert bill analyzer. Analyze the following bill text:

    Bill: {context}


    User Query: {question}

    Based on the user's query, extract and provide the relevant information from the bill. 
    If the requested information is not present in the bill, clearly state that it's not available.
    Present the extracted information in a clear, structured format.
    If there are any unusual or potentially important details related to the query, please mention them.
    

    """
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # Create and return the QA chain
        return load_qa_chain(
            llm=model,
            chain_type="stuff",
            prompt=prompt
        )
