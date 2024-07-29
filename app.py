import streamlit as st
from streamlit.components.v1 import html
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

import os

# Load the Groq API key
groq_api_key = os.getenv('GROQ_API_KEY')

# Funciton to get respone from LLAma 2 model
def getLLamaresponse(input_text):
    # calling llama 3 -8b model with groq
    llm = ChatGroq(temperature=0.01,model="llama3-8b-8192",groq_api_key=st.secrets["groq_api_key"])

    prompt = ChatPromptTemplate.from_template(
    """
        Please analyze the {input_text} and extract the following information:
            * Person's name
            * Company's name they are associate with
            * Person's identity
            * Email address
            * Job role
            * Availability 
            * Reason they are intersted in meeting
        Once you identify this please answer this in a table mentioning "Yes" or "No" for above extarcted information. 
        In case no information is found, please indicate that.
    """)

    # generate the response from llama 2 model
    chain = prompt | llm
    # print(chain)
    response = chain.invoke({"input_text":input_text})
    # print(response.content)

    return response


# Predefined questions and answers
predefined_qa = {
    "What is your company's mission?": "We get people to work - anyone unemployed. We champion full inclusion and upward mobility for the economically disadvantaged.",
    "How can I contact customer support?": "You can contact our customer support via email at https://keelworks.org/contact or call at +1 425.765-2330",
    # "What services do you offer?": "We offer a range of services including consulting, development, and support for various tech solutions.",
    "Where are you located?": "We are located at 2398 West Beach Road, Oak Harbor, WA 98277",
    "What are your business hours?": "Our business hours are Monday to Friday, 9 AM to 5 PM."
}

st.set_page_config(page_title='Keelworks Support Chatbot',
                   layout = 'centered',
                   initial_sidebar_state='collapsed')

st.header("Keelworks Support Chatbot")


# Select a predefined question
question = st.selectbox("Choose a predefined question", list(predefined_qa.keys()))

# Button to display the predefined answer
if st.button('Show Answer'):
    st.write(predefined_qa[question])

# Option to enter a custom query

st.subheader("Or enter your own query")
input_text = st.text_input("Enter your query here")

# Button to get response from LLaMA model
if st.button('Submit'):
    response = getLLamaresponse(input_text)
    st.write(response.content)