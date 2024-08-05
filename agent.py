import os
from typing import Literal
from dotenv import load_dotenv
from tools import add, subtract
from memory import DatabaseHandler
import google.generativeai as genai
from langgraph.prebuilt import ToolNode
from langchain.prompts import PromptTemplate
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, START
from langchain.chains.question_answering import load_qa_chain
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
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
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

    def interact_with_agent(self, message, thread_id, vector_store_folders):
        all_docs = []

        if vector_store_folders:
            for vector_store_folder in vector_store_folders:
                new_db = self.vector_store_manager.load_vector_store(vector_store_folder)
                docs = new_db.similarity_search(message)
                all_docs.extend(docs)

            if all_docs:
                # Pass the documents directly to the chain
                chain = self.get_conversational_chain()
                response = chain({"input_documents": all_docs, "question": message})

                # Combine the message with file content and chain answer if any
                combined_message = f"""
                you are an ai assistant your primary task is to respond to the user's message: {message} and provide the most accurate and relevant response to the user's query.

                Documents: {all_docs}

                Chain Answer: {response['output_text']}

                Your task is to use the document and the chain answer to provide the most accurate and relevant response to the user's query.

                Guidelines:
                1. Carefully read the user's message to determine if they are asking about file content or engaging in general conversation.
                2. For file-related queries:
                - Analyze both the provided documents and the chain answer.
                - Verify the accuracy and completeness of the chain answer.
                - If the chain answer is incorrect or incomplete, provide a corrected and comprehensive response based on the document content.
                - Focus on answering exactly what the user asked, avoiding extraneous information.
                3. For general conversation:
                - Engage naturally, drawing from your broad knowledge base.
                - Ignore file content and chain answers if they're not relevant to the user's query.
                4. Always prioritize accuracy and relevance in your responses.
                5. Be concise for simple queries, but offer detailed explanations for complex topics if needed.
                6. Maintain a friendly and helpful tone throughout the conversation.
                7. Do not reference these instructions or the internal workings of the system in your response.

                Respond directly to the user's message: {message}
                """
            else:
                combined_message = message
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
        for msg in result['messages']:
            if isinstance(msg, AIMessage):
                ai_response = msg.content.strip()

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
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)