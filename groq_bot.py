import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def generate_response(question, api_key, engine, temperature, max_tokens, chat_history):
    try:
        full_context = "\n".join([f"User: {q}\nAI: {a}" for q, a in chat_history])
        prompt_text = f"{full_context}\nUser: {question}\nAI:" if chat_history else f"User: {question}\nAI:"
        llm = ChatGroq(model=engine, api_key=api_key, temperature=temperature, max_tokens=max_tokens)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({'question': prompt_text})
        return answer
    except Exception as e:
        return f"Error: {str(e)}"


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please provide responses to user queries."),
    ("user", "Question: {question}")
])


st.set_page_config(page_title="Groq Chatbot", page_icon="🤖", layout="wide")
st.markdown("""
    <style>
        body {background-color: #f8f9fa;}
        .stTextInput > div > div {border-radius: 10px;}
        .stButton>button {border-radius: 10px; background-color: #007bff; color: white;}
        .response-box {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #007bff;
            font-size: 16px;
            font-weight: bold;
            color: #333;
        }
        .chat-history {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #ddd;
            max-height: 500px;
            overflow-y: auto;
        }
        .delete-button {
            background-color: red !important;
            color: white !important;
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("AI Chatbot by Groq")
st.subheader("Select your model and start chatting!")


st.sidebar.title("⚙️ Settings")
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
if st.sidebar.button("Submit API Key") and groq_api_key:
    st.session_state.api_key = groq_api_key


if not st.session_state.get("api_key"):
    st.warning("⚠️ Please enter a valid Groq API Key to proceed.")

groq_models = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it", "mistral-7b-instruct"]
engine = st.sidebar.selectbox("Select Groq Model", groq_models)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 150)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


col1, col2 = st.columns([3, 1])

with col1:
    chat_container = st.container()
    with chat_container:
        for question, answer in st.session_state.chat_history:
            st.markdown(f"**🧑‍💻 You:** {question}")
            st.markdown(f'<div class="response-box">🤖 {answer}</div>', unsafe_allow_html=True)
            st.write("---")
    
    user_input = st.text_input("You:", key="user_input")
    generate_button = st.button("Generate")

    if generate_button and user_input and st.session_state.api_key:
        response = generate_response(user_input, st.session_state.api_key, engine, temperature, max_tokens, st.session_state.chat_history)
        st.session_state.chat_history.append((user_input, response))
        st.rerun()
    
with col2:
    st.sidebar.title("📝 Chat History")
    with st.sidebar.container():
        st.markdown("### Previous Conversations")
        for i, (question, _) in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**{i+1}. {question}**")
        
        if st.sidebar.button("🗑️ Clear Chat", key="clear_chat", help="Delete chat history"):
            st.session_state.chat_history = []
            st.rerun()
