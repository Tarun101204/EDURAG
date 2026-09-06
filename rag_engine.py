import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Set up HuggingFace (This converts text into numbers locally on your machine)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Tell the system where to save our Chroma database
CHROMA_PATH = "./chroma_db"

# 3. Initialize the database connection globally ONCE, not on every question.
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

def ingest_pdf(file_path: str):
    """This function loads a PDF, chops it up, and saves it to the database."""
    print(f"Reading the PDF: {file_path}...")
    
    # Load Document
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split Document into smaller, readable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Chopped into {len(chunks)} chunks. Saving to database...")

    # Save to ChromaDB
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH)
    print("Successfully saved to database!")
    
    return len(chunks)

def get_answer(query: str, groq_api_key: str):
    """This function searches the database and uses an AI model to answer your question."""
    
    # Use the global db to retrieve the top 3 most relevant chunks
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Initialize the Groq LLM (This is the AI brain that will generate the final answer)
    llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name="openai/gpt-oss-20b",
        temperature=0.2
    )

    # Create the strict instructions for the AI to prevent hallucinations
    system_prompt = (
        "You are a helpful AI assistant for students. Use the following retrieved context to answer the user's question. "
        "If the answer is not in the context, clearly state that you don't know based on the provided document.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Connect the search engine to the AI
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Ask the question and return the answer
    response = rag_chain.invoke({"input": query})
    return response["answer"]
