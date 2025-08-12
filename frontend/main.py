from typing import AsyncGenerator

import streamlit as st
from openai import AsyncOpenAI

from model_config import LocalModel
from settings import refresh_settings

settings = refresh_settings()
aclient = AsyncOpenAI(
    api_key=settings.OLLAMA_API_KEY.get_secret_value(),
    base_url=settings.OLLAMA_URL,
)


st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    async def aresponse_generator() -> AsyncGenerator[str, None]:
        """Generate a response from the assistant asynchronously."""
        stream = await aclient.chat.completions.create(
            model=LocalModel.LLAMA3_2_3B,
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    with st.chat_message("assistant"):
        response = st.write_stream(aresponse_generator())

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
