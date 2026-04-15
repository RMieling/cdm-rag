import os
import time
import uuid

import requests
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from api.utils.logger import frontend_logger

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/api")

# load credentials for Streamlit Authenticator
frontend_logger.info("Loading credentials for Streamlit Authenticator...")
credentials_path = os.path.join(os.getcwd(), "credentials.yaml")
frontend_logger.debug(f"Trying to open credentials file: {credentials_path}")
with open(credentials_path, "r") as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initialize the Authenticator using the loaded config
authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    # config['pre-authorized'],
    auto_hash=True,
)

# Render the Login Widget and handle the Authentication States
try:
    authenticator.login()
except Exception as e:
    st.error(e)

if st.session_state.get("authentication_status") is False:
    st.error("Username/password is incorrect")

elif st.session_state.get("authentication_status") is None:
    st.warning("Please enter your username and password")

elif st.session_state.get("authentication_status"):
    # --- SUCCESSFUL LOGIN ---
    name = st.session_state.get("name")
    username = st.session_state.get("username")
    st.write(f"You are logged in as ID: *{name}*")

    # Page config
    st.set_page_config(page_title="CDM RAG Assistant", page_icon="🔧", layout="wide")
    st.title("CDM RAG Assistant")
    st.caption("Ask questions about the CDM entity defintions, their attributes and relationships.")

    # Check if llm service is ready
    def get_ai_status():
        try:
            response = requests.get(f"{API_URL}/health/ai", timeout=5)
            if response.status_code == 200:
                return response.json().get("status")
        except requests.exceptions.ConnectionError:
            return "unavailable"
        return "error"

    # Check the status on load
    current_status = get_ai_status()

    if current_status == "unavailable":
        st.error("The Backend API is unreachable. Are the Docker containers running?")
        st.stop()

    elif current_status == "downloading":
        # Show a full-page loading message
        with st.spinner(
            "The AI Engine is warming up and downloading models. This might take a few minutes on first boot..."
        ):
            time.sleep(5)
            st.rerun()  # refresh page and check again

    # --- Session State Management ---
    # Generate a unique session ID so the LangGraph memory works correctly
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Chat Interface ---
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display sources if it's an AI message
            if message["role"] == "assistant" and message.get("sources"):
                st.caption(f"**Sources:** {', '.join(message['sources'])}")

    # React to user input
    if prompt := st.chat_input("What can i do for you?"):
        # Add user message to UI state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Ask the Backend
        with st.chat_message("assistant"):
            with st.spinner("Searching data base and thinking..."):
                try:
                    payload = {
                        "question": prompt,
                        "session_id": st.session_state.session_id,
                    }
                    response = requests.post(f"{API_URL}/chat", json=payload, timeout=60)

                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])

                        st.markdown(answer)
                        if sources:
                            st.caption(f"**Sources:** {', '.join(sources)}")

                        # Save AI response to UI state
                        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                    else:
                        st.error("Backend encountered an error.")
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to the backend API. Is it running?")
