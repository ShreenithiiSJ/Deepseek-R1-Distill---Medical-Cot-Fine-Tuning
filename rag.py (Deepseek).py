import streamlit as st  
from langchain_community.document_loaders import PDFPlumberLoader  
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_core.vectorstores import InMemoryVectorStore  
from langchain_ollama import OllamaEmbeddings  
from langchain_core.prompts import ChatPromptTemplate  
from langchain_ollama.llms import OllamaLLM  

st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #FFFFFF; }
.stChatInput input { background-color: #1E1E1E !important; color: #FFFFFF !important; border: 1px solid #3A3A3A !important; }
.stChatMessage:nth-child(odd) { background-color: #1E1E1E; border: 1px solid #3A3A3A; color: #E0E0E0; border-radius: 10px; padding: 15px; margin: 10px 0; }
.stChatMessage:nth-child(even) { background-color: #2A2A2A; border: 1px solid #404040; color: #F0F0F0; border-radius: 10px; padding: 15px; margin: 10px 0; }
.stChatMessage .avatar { background-color: #00FFAA !important; color: #000000 !important; }
.stFileUploader { background-color: #1E1E1E; border: 1px solid #3A3A3A; border-radius: 5px; padding: 15px; }
h1, h2, h3 { color: #00FFAA !important; }
</style>
""", unsafe_allow_html=True)

st.title("DocuMind AI")  
st.markdown("Get smart answers from your documents — fast and clear.")  

PROMPT_TEMPLATE = """
You are a highly skilled medical assistant. Use the provided document context to answer accurately.
Avoid hallucinations. Be short, factual, and clear.

Query: {user_query}  
Context: {document_context}  
Answer:
"""

PDF_STORAGE_PATH = 'document_store/pdfs/'  
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")  
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)  
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")  

def save_uploaded_file(uploaded_file):  
    file_path = PDF_STORAGE_PATH + uploaded_file.name  
    with open(file_path, "wb") as file:  
        file.write(uploaded_file.getbuffer())  
    return file_path  

def load_pdf_documents(file_path):  
    return PDFPlumberLoader(file_path).load()  

def chunk_documents(raw_documents):  
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(raw_documents)  

def index_documents(document_chunks):  
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)  

def find_related_documents(query):  
    return DOCUMENT_VECTOR_DB.similarity_search(query)  

def generate_answer(user_query, context_documents):  
    context_text = "\n\n".join([doc.page_content for doc in context_documents])  
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)  
    return (prompt | LANGUAGE_MODEL).invoke({"user_query": user_query, "document_context": context_text})  

uploaded_pdf = st.file_uploader(
    "Upload a research PDF",
    type="pdf",
    help="Only PDF files. DocuMind will analyze and let you query it."
)

if uploaded_pdf:  
    saved_path = save_uploaded_file(uploaded_pdf)  
    raw_docs = load_pdf_documents(saved_path)  
    chunks = chunk_documents(raw_docs)  
    index_documents(chunks)  
    st.success("Document processed. You can now ask questions about it!")

user_input = st.chat_input("Ask your question about the document...")  
if user_input:  
    with st.chat_message("user"):  
        st.write(user_input)  
    with st.spinner("Thinking..."):  
        relevant_docs = find_related_documents(user_input)  
        answer = generate_answer(user_input, relevant_docs)  
    with st.chat_message("assistant", avatar="🤖"):  
        st.write(answer)

