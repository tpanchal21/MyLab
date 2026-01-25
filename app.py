from flask import Flask, render_template, request

from openai import OpenAI

from os import getenv
from dotenv import load_dotenv
load_dotenv()



app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_chat_response_openai(input)

def get_chat_response_openai(text):
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="<openrouter_api_key>",
    )

    # First API call with reasoning
    response = client.chat.completions.create(
    model="arcee-ai/trinity-mini:free",
    messages=[
            {
                "role": "user",
                "content": text
            }
            ],
    extra_body={"reasoning": {"enabled": True}}
    )

    # Extract the assistant message with reasoning_details
    response = response.choices[0].message

    print(response)
    return response.content




if __name__ == '__main__':
    app.run()
    