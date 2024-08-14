import time

import streamlit as st

from model import gpt2_model


# Streamed response emulator
def response_generator(user_query: str):
    """
    Generates model's response
    Args:
        user_query: user's query

    Returns:

    """
    reply = gpt2_model.predict(text_input=user_query)
    split_strings = reply.split(' ')
    for word in split_strings:
        yield word + ' '
        time.sleep(0.05)


def greeting():
    text = ('Hello!  \n'
            'what is your prompt?')
    return text


def write_stream(stream):
    """
    Showed text as a stream
    Args:
        stream: input text

    Returns:

    """
    result = ""
    container = st.empty()
    for chunk in stream:
        result += chunk
        container.write(result, unsafe_allow_html=True)
    return result


st.title("SuperBot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": greeting()})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# React to user input
if prompt := st.chat_input("Navigate to a different URL after 5 seconds when a key is pressed"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = write_stream(response_generator(user_query=prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
