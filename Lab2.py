import streamlit as st
from openai import OpenAI
import fitz

def read_pdf(uploaded_file):
    pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf_doc:
        text += page.get_text()
    return text

st.title("Lab 2 - Document Summarizer")

openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

# Sidebar: summary type options
summary_type = st.sidebar.radio(
    "Choose a summary type:",
    (
        "Summarize the document in 100 words",
        "Summarize the document in 2 connecting paragraphs",
        "Summarize the document in 5 bullet points",
    ),
)

# Sidebar: model choice
use_advanced = st.sidebar.checkbox("Use advanced model")
if use_advanced:
    model = st.sidebar.selectbox("Advanced model:", ("gpt-5.4", "o3"))
else:
    model = "gpt-5-nano"

uploaded_file = st.file_uploader(
    "Upload a document (.txt or .pdf)", type=("txt", "pdf")
)

if uploaded_file:
    file_extension = uploaded_file.name.split('.')[-1]
    if file_extension == 'txt':
        document = uploaded_file.read().decode()
    elif file_extension == 'pdf':
        document = read_pdf(uploaded_file)
    else:
        st.error("Unsupported file type.")
        st.stop()

    messages = [
        {
            "role": "user",
            "content": f"Here's a document: {document} \n\n---\n\n {summary_type}.",
        }
    ]

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    st.write_stream(stream)