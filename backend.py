from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
import asyncio

load_dotenv()

# Request model for chat endpoint
class ChatRequest(BaseModel):
    question: str

rag_chain = None
api_key = None
model = None

def initialize_rag_chain():
    global rag_chain, api_key, model
    
    try:
        # Load API key
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        print("✅ API key loaded successfully")
        
        # Initialize model
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=api_key)
        print("✅ Model initialized successfully")
        
        # Check if data directory exists
        data_path = 'data/portfolioData'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data directory {data_path} not found")
        
        print(f"✅ Data directory found: {data_path}")
        
        # Load documents
        loader = DirectoryLoader(data_path, glob='**/context.txt', loader_cls=TextLoader)
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} documents")
        
        # Initialize embeddings and vectorstore
        embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", api_key=api_key)
        vectorstore = Chroma.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        print("✅ Vector store and retriever created successfully")
        
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_template(
            "Act as me(Manikanta) and Use the following context to answer the question:\n\n"
            "{context}\n\n"
            "Question: {input}\n"
            "Answer:"
        )
        print("✅ Prompt template created successfully")
        
        # Create document chain and RAG chain
        document_chain = create_stuff_documents_chain(llm=model, prompt=prompt_template)
        rag_chain = create_retrieval_chain(retriever, document_chain)
        print("✅ RAG chain created successfully")
        
        print("🎉 RAG chain initialization completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during startup: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def startup_event():
    """Synchronous startup event for FastAPI"""
    success = initialize_rag_chain()
    if not success:
        print("❌ Failed to initialize RAG chain during startup")
    else:
        print("✅ Startup event completed successfully")

app = FastAPI(on_startup=[startup_event])

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://manikantasai55555.github.io", 
        "http://localhost:5173"
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global rag_chain
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG chain not initialized.")
    try:
        response = rag_chain.invoke({"input": request.question})
        return {"answer": response["answer"]}
    except Exception as e:
        print(f"❌ Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    
@app.get("/health")
async def health_check():
    return {"status": "ok", "rag_chain_initialized": rag_chain is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
