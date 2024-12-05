import json
from langchain_openai import ChatOpenAI

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader

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

def retrieveLlmResp(question, vectorStore):
    llm = ChatOpenAI(
            model="gpt-3.5-turbo-1106",
            temperature=0.4
        )
    from operator import itemgetter
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    Think step by step before providing a detailed answer.
    <context>
    {context}
    </context>

    Question: {input}""")
    
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

    retriever=vectorStore.as_retriever()
    document_chain=create_stuff_documents_chain(llm,prompt)
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    response = retrieval_chain.invoke({"input": question})
    return {"documents": response['answer'], "question": response['input']}

vectorStore = None

def lambda_handler(event, context):
    # TODO implement

    question = event.get("question", None)
    if not question:
        question = "no question asked."
    docs = get_documents_from_pdf("Resume.pdf")
    vectorStore = create_db(docs)
    
    response = retrieveLlmResp(question, vectorStore)

        # Create a response
    response = {
            'statusCode': 200,
            'body': json.dumps({
                'received_question': question,
                'message': response
            })
        }
    return response

if __name__ == "__main__":
    print(lambda_handler(event={"question":"which company did he work in year 2022?"}, context=any))