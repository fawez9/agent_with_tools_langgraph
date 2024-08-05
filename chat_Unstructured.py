import re
import os
import json
import pandas as pd
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader

class VectorStoreManager:
    def __init__(self):
        self.vector_store_folder = None

    def create_vector_store(self, text_chunks,id):
        unique_id = id
        self.vector_store_folder = f"faiss_index_{unique_id}"
        os.makedirs(self.vector_store_folder, exist_ok=True)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create metadata for each chunk
        metadatas = [{"chunk": i, "total_chunks": len(text_chunks)} for i in range(len(text_chunks))]
        
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas)
        vector_store.save_local(self.vector_store_folder)
        return self.vector_store_folder

    def load_vector_store(self, folder):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)


class EmbeddingManager:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)

    def process_files_and_url(self, uploaded_files, url):
        raw_text = self.get_all_text_from_files(uploaded_files)
        if url:
            raw_text += self.get_url_text(url)
        text_chunks = self.get_text_chunks(raw_text)
        return text_chunks

    def get_all_text_from_files(self, uploaded_files):
        raw_text = ""
        print(uploaded_files)
        pdf_docs = [file for file in uploaded_files if file.endswith('.pdf')]
        csv_docs = [file for file in uploaded_files if file.endswith('.csv')]
        txt_docs = [file for file in uploaded_files if file.endswith('.txt')]
        xls_docs = [file for file in uploaded_files if file.endswith('.xls')]
        json_docs = [file for file in uploaded_files if file.endswith('.json')]

        if pdf_docs:
            raw_text += self.get_pdf_text(pdf_docs)
        if csv_docs:
            raw_text += self.get_csv_text(csv_docs)
        if txt_docs:
            raw_text += self.get_txt_text(txt_docs)
        if xls_docs:
            raw_text += self.get_xls_text(xls_docs)
        if json_docs:
            raw_text += self.get_json_text(json_docs)

        return raw_text

    def preprocess_text(self, text, file_type):
        # Common preprocessing
        text = text.replace('\n', ' ').replace('\r', '')
        
        if file_type == 'pdf':
            # PDF-specific preprocessing
            text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        elif file_type == 'csv' or file_type == 'xls':
            # Spreadsheet-specific preprocessing
            text = re.sub(r'\s{2,}', ' | ', text)  # Replace multiple spaces with a delimiter
        elif file_type == 'json':
            # JSON-specific preprocessing
            text = re.sub(r'[{}\[\]]', '', text)  # Remove brackets
        
        return text

    # Update the get_*_text methods to use this preprocessing
    def get_pdf_text(self, pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += self.preprocess_text(page.extract_text(), 'pdf') + "\n"
        return text

    # Similarly, update other get_*_text methods

    def get_csv_text(self, csv_docs):
        text = ""
        for csv in csv_docs:
            df = pd.read_csv(csv)
            text += df.to_string(index=False)
        return text

    def get_txt_text(self, txt_docs):
        text = ""
        for txt in txt_docs:
            text += txt.read().decode("utf-8")
        return text

    def get_xls_text(self, xls_docs):
        text = ""
        for xls in xls_docs:
            df = pd.read_excel(xls)
            text += df.to_string(index=False)
        return text

    def get_json_text(self, json_docs):
        text = ""
        for json_file in json_docs:
            data = json.load(json_file)
            text += json.dumps(data, indent=2)
        return text

    def get_url_text(self, url):
        loader = UnstructuredURLLoader(urls=[url])
        documents = loader.load()
        text = "\n".join([doc.page_content for doc in documents])
        return text

    def get_text_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", ", ", " "],
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunks[i] = f"Chunk {i+1} of {len(chunks)}:\n{chunk}"
        
        return chunks



