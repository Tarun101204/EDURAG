import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# Restored to langchain_classic (Your original code was right!)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

CHROMA_PATH = "./chroma_db"
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

def ingest_pdf(file_path: str):
    print(f"Reading the PDF: {file_path}...")
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Chopped into {len(chunks)} chunks. Saving to database...")

    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH)
    print("Successfully saved to database!")
    
    return len(chunks)

def get_answer(query: str, groq_api_key: str):
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name="openai/gpt-oss-20b",
        temperature=0.2
    )

    system_prompt = (
        "You are a helpful AI assistant for students. Use the following retrieved context to answer the user's question. "
        "If the answer is not in the context, clearly state that you don't know based on the provided document.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    return response["answer"]