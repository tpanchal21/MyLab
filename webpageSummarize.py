from flask import Flask, request, redirect, session
from flask_cors import CORS
#from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import requests
from bs4 import BeautifulSoup

import stripe
# This is your test secret API key.
stripe.api_key = 'sk_test_51QWgpwF34Mmm3mipmecR5vqLalAcNVsbpKAPeKonfeZk4fIjgbxQ6fJ3s43C3kQkgTYG79h1KoyoON94zbKqKniC00miSFmDy1'

chat_advance = Flask(__name__,
            static_url_path='',
            static_folder='public')

YOUR_DOMAIN = 'http://localhost:8080/'
#chat_advance = Flask(__name__)
CORS(chat_advance)  # Enable CORS for all routes and origins
chat_advance.secret_key = "chatbotusinglanggraphlangchain"
#load_dotenv(dotenv_path=".env", override=True)

# Initialize the ChatOpenAI model
chat = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key="your-openai-api-key")  # Replace with your key

from openai import OpenAI

# Set your secret key. Remember to switch to your live secret key in production.
# See your keys here: https://dashboard.stripe.com/apikeys
stripe.api_key = '<stripe key>'

@chat_advance.route('/fulfill-checkout', methods=['GET'])
def fulfill_checkout():
    session_id = session['checkout_session_id']
    print("Fulfilling Checkout Session ", session_id)

    # TODO: Make this function safe to run multiple times,
    # even concurrently, with the same session ID

    # TODO: Make sure fulfillment hasn't already been
    # peformed for this Checkout Session

    # Retrieve the Checkout Session from the API with line_items expanded
    checkout_session = stripe.checkout.Session.retrieve(
        session_id,
        expand=['line_items'],
    )

    print(checkout_session)
    # Check the Checkout Session's payment_status property
    # to determine if fulfillment should be peformed
    if checkout_session.payment_status != 'unpaid':
        return checkout_session
    else:
        return "payment still pending. refresh it again."
        # TODO: Perform fulfillment of the line items

        # TODO: Record/save fulfillment status for this
        # Checkout Session


@chat_advance.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    try:
        checkout_session = stripe.checkout.Session.create(
            line_items=[
                {
                    # Provide the exact Price ID (for example, pr_1234) of the product you want to sell
                    'price': 'price_1QXRUpF34Mmm3mip2RWWX6DK',
                    'quantity': 1,
                },
            ],
            mode='subscription',
            success_url=YOUR_DOMAIN + '/fulfill-checkout',
            cancel_url=YOUR_DOMAIN + '/cancel.html',
        )
    except Exception as e:
        return str(e)
    print(" checkout session id ",checkout_session.id)
    session['checkout_session_id'] = checkout_session.id
    return redirect(checkout_session.url, code=303)

@chat_advance.route("/fetch", methods=["GET", "POST"])
def fetch_webpage_text():
    """
    Fetch and extract the text content of a given HTTPS webpage.
    Args:
        url (str): The URL of the webpage.
    Returns:
        str: The text content of the webpage.
    """
    try:
        
        # Send a GET request to the URL
        url = request.args.get('url', default='hi')
        email = request.args.get('email', default = 'lala')
        
        # Example usage
        statuses = get_payment_status_by_email(email)
        if statuses == "No customer found with email" or statuses == "No payment intents found":
            return "No payment found for customer. Please buy the extension."
        
        print(statuses);
        
        # Simulate a browser with headers
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for HTTP issues
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract and return only the text
        #print(soup)
        htmlContent = soup.get_text(strip=True)
        return get(htmlContent)
    except Exception as e:
        print(f"Error fetching the webpage: {e}")
        return None

#@chat_advance.route("/get", methods=["GET", "POST"])
def get(msg):
    #msg = request.args.get('msg', default='hi')

    try:

        client = OpenAI()

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You have to summarize user's message in 5 bullet points."},
                {
                    "role": "user",
                    "content": msg
                }
            ]
        )

        #print(completion)
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error invoking ChatOpenAI: {e}")
        return None
    

def get_payment_status_by_email(email):
    try:
        # Step 1: Retrieve the customer by email
        customers = stripe.Customer.list(email=email)
        
        if not customers.data:
            return f"No customer found with email"

        customer = customers.data[0]  # Assuming one customer per email
        customer_id = customer["id"]

        # Step 2: Retrieve payment intents for the customer
        payment_intents = stripe.PaymentIntent.list(customer=customer_id)

        if not payment_intents.data:
            return f"No payment intents found"

        # Step 3: Gather payment statuses
        payment_statuses = []
        for intent in payment_intents.data:
            payment_statuses.append({
                "payment_intent_id": intent["id"],
                "amount": intent["amount"] / 100,  # Amount in dollars
                "currency": intent["currency"],
                "status": intent["status"]
            })

        return payment_statuses

    except stripe.error.StripeError as e:
        return f"Stripe API error: {e.user_message}"
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    # HTTPS
    #from werkzeug.serving import make_server
    #https_server = make_server('0.0.0.0', 5443, chat_advance, ssl_context=('cert.pem', 'key.pem'))
    #print("HTTPS Server running on port 5443")
    #https_server.serve_forever()

    #http
    chat_advance.run(debug=False, port=8080, host="0.0.0.0")
    