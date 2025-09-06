

# app.py

import streamlit as st
from agent_setup import setup_agent_executor

st.title("🤖 NL2SQL Chatbot for MySQL")
st.write("Ask questions about your database in plain English. I'll translate it to SQL and get the answer for you!")

# Initialize the agent executor in session state
if "agent_executor" not in st.session_state:
    with st.spinner("Initializing the AI agent... Please wait."):
        # The setup function is called only once and the result is stored
        st.session_state.agent_executor = setup_agent_executor()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get the assistant's response
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.agent_executor.invoke({"input": prompt})
            output = response["output"]
        except Exception as e:
            output = f"An error occurred: {e}"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(output)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": output})

