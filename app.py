from flask import Flask, request, session, render_template
import chromadb
from openai import OpenAI
import markdown
from os import getenv
import json
import os
import requests

import uuid
from dotenv import load_dotenv

from pypdf import PdfReader

load_dotenv()

# -----------------------
# OpenAI client
# -----------------------
client = OpenAI(
base_url="https://openrouter.ai/api/v1"
)

#client = OpenAI()
# -----------------------
# ChromaDB setup (persistent)
# -----------------------
chroma_client = chromadb.Client(
    chromadb.config.Settings(
        persist_directory="./chroma_db"
    )
)

collection = chroma_client.get_or_create_collection(
    name="rag_docs"
)


# -----------------------
# Flask setup
# -----------------------
app = Flask(__name__)
app.secret_key = "replace-this-in-prod"


def get_documents_from_pdf(filenames, chunk_size=500, overlap=100):
    documents = []

    for file in (x.strip() for x in filenames.split(",")):
        if file.lower().endswith(".pdf"):
            #pdf_path = os.path.join(folder_path, file)
            reader = PdfReader(file)

            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

            # ---- Chunk text
            start = 0
            while start < len(full_text):
                end = start + chunk_size
                chunk = full_text[start:end]

                documents.append({
                    "text": chunk,
                    "source": file
                })

                start = end - overlap

        return documents

#DOCUMENTS = get_documents_from_pdf("Resume.pdf")


# -----------------------
# Load documents once
# -----------------------
if collection.count() == 0:
    pdf_chunks = get_documents_from_pdf("Resume-AI.pdf")

    collection.add(
        documents=[c["text"] for c in pdf_chunks],
        metadatas=[{"source": c["source"]} for c in pdf_chunks],
        ids=[f"pdf-{i}" for i in range(len(pdf_chunks))]
    )

# -----------------------
# Chat history helpers
# -----------------------
def get_chat_history():
    if "chat_id" not in session:
        session["chat_id"] = str(uuid.uuid4())
        session["history"] = []
    return session["history"]


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_chat_response_openai(input)

def write_session_history_to_file(filename, history):
    """
    Write session history to a JSON file.

    :param filename: path to file (e.g. sessions/abc123.json)
    :param history: session["history"] list
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            {"history": history},
            f,
            indent=2,
            ensure_ascii=False
        )

def is_about_tushar(text):
    """
    Check if the question is about Tushar Panchal.
    """
    keywords = ["tushar", "panchal", "he", "him", "his", "experience", "background"]
    return any(keyword.lower() in text.lower() for keyword in keywords)


def get_wiki_answer(text):
    """
    Fetch answer from Wikipedia.
    """
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "titles": text,
                "prop": "extracts",
                "explaintext": True
            }
        )
        data = response.json()
        pages = data["query"]["pages"]
        page = next(iter(pages.values()))
        return page.get("extract", "No information found on Wikipedia.")
    except Exception as e:
        return f"Could not retrieve from Wikipedia: {str(e)}"


def wiki_answer(text):
     ### Working With Tools
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_community.tools import WikipediaQueryRun

    ## wikipedia Tools
    api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=500)
    wiki=WikipediaQueryRun(api_wrapper=api_wrapper)


    # Wiki search
    docs = wiki.invoke({"query": text})
    return docs

def get_chat_response_openai(text):
    resp = ""
    try:
        history = get_chat_history()

        # Check if question is about Tushar
        about_tushar = is_about_tushar(text)
        print ("about tushar: ", about_tushar)
        if about_tushar:
            # Retrieve from Chroma for Tushar-related questions
            results = collection.query(
                query_texts=[text],
                n_results=5
            )
            retrieved_docs = results["documents"][0]
            context = "\n".join(retrieved_docs)
        else:
            # Retrieve from Wikipedia for other questions
            context = wiki_answer(text)
        
        #return context
        # ---- Build messages (history aware)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Tushar's RAG based AI Assistant. Answer all salutations. "
                    "Use the provided context to answer the question. "
                    "If the answer is not found in the context, look in chat history otherwise, say 'I do not know the answer to that. I can answer questions about Tushar Panchal.'."
                )
            }
        ]

        # Add conversation history
        for turn in history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        # Current question
        messages.append({
            "role": "user",
            "content": f"""
                Context:
                {context}

                Question:
                {text}
                """
                    })

        # First API call with reasoning
        response = client.chat.completions.create(
        model= "arcee-ai/trinity-mini:free",
        messages=messages,
        temperature=0
        )

        # Extract the assistant message with reasoning_details
        resp = response.choices[0].message.content

        return markdown.markdown(resp)
    except Exception as e:
        resp= f"<b>Error occured:</b> {str(e)}"
        return resp
    finally:
        # Save history
        history.append({
            "user": text,
            "assistant": resp
        })
        session["history"] = history
        write_session_history_to_file(f"sessions/{session["chat_id"]}", history)
 


if __name__ == '__main__':
    app.run()
