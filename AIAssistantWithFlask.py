from flask import Flask, render_template, request

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs

def get_documents_from_pdf(filepath):
    pdf_files = [filepath]

    all_pages = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)  # Use the variable pdf_file directly
        pages = loader.load_and_split()
        all_pages.extend(pages)
    
    return all_pages

def create_db(docs):
    vectorStore = Chroma.from_documents(
        documents=docs,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(),
    )
    return vectorStore

def create_chain(vectorStore):
    model = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        temperature=0.4
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    # chain = prompt | model
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    # Replace retriever with history aware retriever
    retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        # retriever, Replace with History Aware Retriever
        history_aware_retriever,
        chain
    )

    return retrieval_chain

def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "chat_history": chat_history,
        "input": question,
    })
    return response["answer"]

chat_advance = Flask(__name__)

@chat_advance.route("/")
def index():
    return render_template('chat.html')


@chat_advance.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

chat_history = []
def get_Chat_response(text):
    docs = get_documents_from_pdf("Election2024.pdf")
    vectorStore = create_db(docs)
    chain = create_chain(vectorStore)
    response = process_chat(chain, text, chat_history)
    # Initialize chat history

    chat_history.append(HumanMessage(content=text))
    chat_history.append(AIMessage(content=response))
    return response

if __name__ == '__main__':
    chat_advance.run(debug=True, port=5000, host="0.0.0.0")
    