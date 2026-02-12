from flask import Flask, render_template, request, session
from dotenv import load_dotenv
import markdown
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
#from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_community.document_loaders import WebBaseLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
### Working With Tools
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun

from typing import Annotated, List

from typing_extensions import TypedDict
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic.v1 import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages

import os, uuid

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
pdf_files = []
def get_documents_from_pdf(folder_path):
    # Get a list of all files in the specified folder
    filenames = os.listdir(folder_path)
    
    # Filter to include only files (excluding directories)
    #pdf_files = [f for f in filenames if os.path.isfile(os.path.join(folder_path, f))]

    all_pages = []
    for pdf_file in filenames:
        loader = PyPDFLoader(folder_path+"/"+pdf_file)  # Use the variable pdf_file directly
        pages = loader.load_and_split()
        all_pages.extend(pages)

    return all_pages


def retrieve(state):

    from operator import itemgetter
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template("""
    You are Tushar Panchal's AI Assistant. Answer the questions based only on the provided context about {contextfiles}.
    Think step by step before providing a detailed answer.
    <context>
    {context}
    </context>

    Question: {input}""")
    question = state["messages"][-1].content

    retriever=vectorStore.as_retriever()
    document_chain=create_stuff_documents_chain(llm,prompt)
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    # Get a list of all files in the specified folder
    filenames = os.listdir(UPLOAD_FOLDER)
    
    response = retrieval_chain.invoke({"input": question, "contextfiles": ', '.join(filenames)})
    state["messages"].append(AIMessage(response['answer']))
    #response = retrieval_chain.invoke({"input": question, "contextfiles": 'technologycover, resume'})
    #return {"documents": response['answer'], "question": question}
    return {"messages":state["messages"]}
 
def wiki_search(state):
    """
    wiki search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    
    question = question = state["messages"][-1].content

    # if "wiki search" not in question:
    #     return {"messages":llm.invoke(state["messages"])}
    
    ## wikipedia Tools
    api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=500)
    wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

    # Wiki search
    docs = wiki.invoke({"query": question})
    print("---WIKI SEARCH RESULTS---",docs)
    #print(docs["summary"])
    wiki_results = docs
    #wiki_results = page_content=wiki_results)
    state["messages"].append(AIMessage(wiki_results))
    return {"messages" : state["messages"]}
    """

    #response = llm.invoke(state["messages"][-1].content)

    #return {"documents": response.content, "question": state["messages"][-1].content}
    return {"messages":llm.invoke(state["messages"])}
    """


### Edges ###
def route_question(state):
    """
    Route question to wiki search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["messages"][-1].content
    source = question_router(question)
    if source.datasource == "wiki_search":
        print("---ROUTE QUESTION TO Wiki SEARCH---")
        return "wiki_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia or a vectorstore.",
    )

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


from langgraph.graph import END, StateGraph, START


def question_router(question):
    # LLM with function call
    from langchain_openai import ChatOpenAI
    import os
    #groq_api_key=userdata.get('groq_api_key')
    #os.environ["GROQ_API_KEY"]=groq_api_key
    structured_llm_router = llm.with_structured_output(RouteQuery)

    # Get a list of all files in the specified folder
    filenames = os.listdir(UPLOAD_FOLDER)
    commaDelimitedFiles = ". ".join(filenames)
    # Prompt
    #system = """You are an expert at routing a user question to a vectorstore or wikipedia.
    #Use wiki-search if it is specifically asked for. Otherwise use vectorstore."""

    system = f"""You are an expert at routing a user question to a vectorstore or wikipedia.
    The vectorstore contains documents related to Tushar Panchal and {commaDelimitedFiles}.
    Use the vectorstore for questions on these topics. Otherwise, use wiki-search."""

    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    question_router = route_prompt | structured_llm_router
    return question_router.invoke({"question": question})

chat_advance = Flask(__name__)
chat_advance.secret_key = "chatbotusinglanggraphlangchain"
load_dotenv()

@chat_advance.route("/")
def index():
    return render_template('chat.html')

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", api_key="AIzaSyAV2lp4BtzUz4VkI42acFlxvYs7nacTBMY")

vectorStore = Chroma.from_documents(
    documents=get_documents_from_pdf("uploads"),
    collection_name="rag-chroma",
    embedding=embeddings
)

@chat_advance.route("/sync", methods=["GET", "POST"])
def sync():
    vectorStore = Chroma.from_documents(
        documents=get_documents_from_pdf("uploads"),
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(),
    )
    return "success"

@chat_advance.route('/upload', methods=['POST','GET'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and file.filename.endswith('.pdf'):
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
        vectorStore = Chroma.from_documents(
            documents=get_documents_from_pdf("uploads"),
            collection_name="rag-chroma",
            embedding=OpenAIEmbeddings(),
            )
        return 'File uploaded and vectorstore created successfully!'
    else:
        return 'Only PDF files are allowed.'

@chat_advance.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):
    if 'thread_id' not in session:
        session['thread_id'] = str(uuid.uuid4())  # Generate a unique ID
        
    # Initialize chat history
    #response = app.invoke({"question": text}, {"configurable": {"thread_id": "42"}})
    response = app.invoke({'messages':("user",text)},  {"configurable": {"thread_id": session['thread_id']}})

    return  markdown.markdown(response['messages'][-1].content)

class State(TypedDict):
  # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
  messages:Annotated[list,add_messages]
  
def chatbot(state:State):
    return {"messages":llm.invoke(state['messages'])}

if __name__ == '__main__':
    #docs = get_documents_from_pdf("Election2024.pdf")
    workflow = StateGraph(State)

    #workflow = StateGraph(GraphState)

        # Define the nodes
    workflow.add_node("wiki_search", wiki_search)  # web search
    workflow.add_node("retrieve", retrieve)  # retrieve

    # Build graph
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "wiki_search": "wiki_search",
            "vectorstore": "retrieve",
        },
    )

    workflow.add_edge( "retrieve", END)
    workflow.add_edge( "wiki_search", END)
    """
    workflow.add_node("chatbot",chatbot)


    workflow.add_edge(START,"chatbot")
    workflow.add_edge("chatbot",END)
    """

    # Compile
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    

    # llm = ChatOpenAI(
    #     model="gpt-3.5-turbo-1106",
    #     temperature=0.4
    # )
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-3-flash-preview", 
        temperature=0.4,
        api_key="AIzaSyAV2lp4BtzUz4VkI42acFlxvYs7nacTBMY"
    )
    chat_advance.run(debug=True, port=5000, host="0.0.0.0")
    