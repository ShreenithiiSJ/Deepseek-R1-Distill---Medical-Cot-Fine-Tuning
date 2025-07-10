import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

st.markdown("""
<style>
    .main { background-color: #1a1a1a; color: #ffffff; }
    .sidebar .sidebar-content { background-color: #2d2d2d; }
    .stTextInput textarea { color: #ffffff !important; }
    .stSelectbox div[data-baseweb="select"],
    .stSelectbox option,
    .stSelectbox svg,
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
        fill: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Code Mentor")
st.caption("👩‍💻 Built to help you debug, explain, and write better code — faster.")

with st.sidebar:
    st.header("🛠️ Settings")
    selected_model = st.selectbox("Choose Your Model", ["deepseek-r1:1.5b", "deepseek-r1:3b"], index=0)
    st.divider()
    st.markdown("### What I Can Do")
    st.markdown("""
    - 🧠 Explain Python code
    - 🔍 Help debug tricky bugs
    - ✍️ Write docstrings or full functions
    - 💡 Suggest better logic
    - ⚡ Optimize code performance
    """)
    st.divider()
    st.markdown("Made with ❤️ using [LangChain](https://python.langchain.com) + [Ollama](https://ollama.ai)")

llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.3
)

system_prompt = SystemMessagePromptTemplate.from_template(
    "You're a helpful and smart AI coding mentor. Be direct, suggest improvements, and always explain with code if needed."
)

if "message_log" not in st.session_state:
    st.session_state.message_log = [
        {"role": "ai", "content": "Hey there! I'm your AI Code Mentor. Ask me anything about code 👨‍💻"}
    ]

chat_container = st.container()

with chat_container:
    for msg in st.session_state.message_log:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

user_query = st.chat_input("What's your coding issue today?")

def generate_ai_response(prompt_chain):
    chain = prompt_chain | llm_engine | StrOutputParser()
    return chain.invoke({})

def build_prompt_chain():
    sequence = [system_prompt]
    for m in st.session_state.message_log:
        if m["role"] == "user":
            sequence.append(HumanMessagePromptTemplate.from_template(m["content"]))
        elif m["role"] == "ai":
            sequence.append(AIMessagePromptTemplate.from_template(m["content"]))
    return ChatPromptTemplate.from_messages(sequence)

if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    with st.spinner("💬 Thinking..."):
        chain = build_prompt_chain()
        reply = generate_ai_response(chain)
    st.session_state.message_log.append({"role": "ai", "content": reply})
    st.rerun()
