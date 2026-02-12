import os
import uuid
import markdown
from typing import Annotated, Literal
from typing_extensions import TypedDict

from flask import Flask, render_template, request, session
from dotenv import load_dotenv
from pydantic.v1 import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Constants
UPLOAD_FOLDER = 'uploads'
SECRET_KEY = "chatbotusinglanggraphlangchain"
GOOGLE_API_KEY = "AIzaSyAV2lp4BtzUz4VkI42acFlxvYs7nacTBMY"
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "models/gemini-3-flash-preview"
COLLECTION_NAME = "rag-chroma"

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Data Models
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia or a vectorstore.",
    )

class State(TypedDict):
    """Graph state with message history."""
    messages: Annotated[list, add_messages]

# Flask App
chat_advance = Flask(__name__)
chat_advance.secret_key = SECRET_KEY
load_dotenv()

# Global Variables
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0.4,
    api_key=GOOGLE_API_KEY
)
vectorStore = None

# Utility Functions
def get_documents_from_pdf(folder_path):
    """Load documents from all PDFs in folder."""
    all_pages = []
    for pdf_file in os.listdir(folder_path):
        if pdf_file.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(folder_path, pdf_file))
            pages = loader.load_and_split()
            all_pages.extend(pages)
    return all_pages

def init_vectorstore():
    """Initialize vector store from PDFs."""
    global vectorStore
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=GOOGLE_API_KEY
    )
    documents = get_documents_from_pdf(UPLOAD_FOLDER)
    vectorStore = Chroma.from_documents(
        documents=documents,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

def question_router(question):
    """Route question to vectorstore or wiki search."""
    structured_llm_router = llm.with_structured_output(RouteQuery)
    filenames = os.listdir(UPLOAD_FOLDER)
    comma_delimited = ", ".join(filenames) if filenames else "None"
    
    system = f"""You are an expert at routing a user question to a vectorstore or wikipedia.
The vectorstore contains documents related to Tushar Panchal and {comma_delimited}.
Use the vectorstore for questions on these topics. Otherwise, use wiki-search."""
    
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])
    
    question_router_chain = route_prompt | structured_llm_router
    return question_router_chain.invoke({"question": question})

# Node Functions
def retrieve(state):
    """Retrieve from vectorstore and generate response."""
    question = state["messages"][-1].content
    filenames = os.listdir(UPLOAD_FOLDER)
    
    prompt = ChatPromptTemplate.from_template("""
You are Tushar Panchal's AI Assistant. Answer the questions based only on the provided context about {contextfiles}.
Think step by step before providing a detailed answer.
<context>
{context}
</context>

Question: {input}""")
    
    retriever = vectorStore.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({
        "input": question,
        "contextfiles": ', '.join(filenames) if filenames else "None"
    })
    
    state["messages"].append(AIMessage(response['answer']))
    return {"messages": state["messages"]}

def wiki_search(state):
    """Search Wikipedia for relevant information."""
    question = state["messages"][-1].content
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
    docs = wiki.invoke({"query": question})
    
    state["messages"].append(AIMessage(docs))
    return {"messages": state["messages"]}

def route_question(state):
    """Conditional edge to route question."""
    question = state["messages"][-1].content
    source = question_router(question)
    return "wiki_search" if source.datasource == "wiki_search" else "vectorstore"

# Flask Routes
@chat_advance.route("/")
def index():
    return render_template('chat.html')

@chat_advance.route("/sync", methods=["GET", "POST"])
def sync():
    try:
        init_vectorstore()
        return "success"
    except Exception as e:
        return f"error: {str(e)}", 500

@chat_advance.route('/upload', methods=['POST', 'GET'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    
    if not file.filename.endswith('.pdf'):
        return 'Only PDF files are allowed.', 400
    
    file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    init_vectorstore()
    return 'File uploaded and vectorstore created successfully!'


@chat_advance.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Error: Empty message", 400
    return get_Chat_response(msg)

def get_Chat_response(text):
    """Generate chat response using the graph."""
    if 'thread_id' not in session:
        session['thread_id'] = str(uuid.uuid4())
    
    response = app.invoke(
        {'messages': [HumanMessage(content=text)]},
        {"configurable": {"thread_id": session['thread_id']}})
    
    return markdown.markdown(response['messages'][-1].content)

if __name__ == '__main__':
    # Initialize vector store
    init_vectorstore()
    
    # Build workflow
    workflow = StateGraph(State)
    workflow.add_node("wiki_search", wiki_search)
    workflow.add_node("retrieve", retrieve)
    
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "wiki_search": "wiki_search",
            "vectorstore": "retrieve",
        },
    )
    
    workflow.add_edge("retrieve", END)
    workflow.add_edge("wiki_search", END)
    
    # Compile app
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    
    # Run Flask
    chat_advance.run(debug=True, port=5000, host="0.0.0.0")
