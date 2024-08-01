import streamlit as st

from agent import generate_response
from utils import write_message

# Page Config
st.set_page_config("Turkish Recipe Chatbot", page_icon=":shallow_pan_of_food:")

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm a chatbot knowledgable about Turkish food recipes. How can I help you?"},
    ]

# Submit handler
def handle_submit(message):
    """
    Submit handler. Generates the response using the LangChain 
    agent and writes back the message to the chat Session State.
    """

    # Handle the response
    with st.spinner('Thinking...'):
        # Call the agent
        response = generate_response(message)
        write_message("assistant", response)


# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if question := st.chat_input("What is up?"):
    # Display user message in chat message container
    write_message('user', question)

    # Generate a response
    handle_submit(question)
