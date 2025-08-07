from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough


class PDFRAG:
    def __init__(self, file_path: str, llm):
        self.file_path = file_path
        self.llm = llm

    def load_documents(self):
        # Step 1.Load Documents
        loader = PyMuPDFLoader(self.file_path)
        docs = loader.load()
        return docs

    def split_documents(self, docs):
        # Step 2. Split Documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_documents = text_splitter.split_documents(docs)
        return split_documents

    def create_vectorstore(self, split_documents):
        # Embedding
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Step 3. Create DB
        vectorstore = FAISS.from_documents(
            documents=split_documents, embedding=embeddings
        )
        return vectorstore

    def create_retriever(self):
        vectorstore = self.create_vectorstore(
            self.split_documents(self.load_documents())
        )
        # Step 4. Create Retriever
        retriever = vectorstore.as_retriever()
        return retriever

    def create_chain(self, retriever):
        # Step 5. Create Prompt
        prompt = PromptTemplate.from_template(
            """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 

        #Context: 
        {context}

        #Question:
        {question}

        #Answer:"""
        )

        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
