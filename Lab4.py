# Streamlit Cloud fix: Chroma needs a newer sqlite3 than the Cloud has.
# pysqlite3-binary only installs on Linux (see requirements.txt), so on the
# Mac this import fails and we just keep the normal sqlite3.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import time
import streamlit as st
from openai import OpenAI, PermissionDeniedError
import fitz
import chromadb

st.title("Lab 4 - Course Information Chatbot (RAG)")

openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

# The 7 course PDFs live in Lab4_PDFs/ next to this file
PDF_FOLDER = os.path.join(os.path.dirname(__file__), "Lab4_PDFs")
EMBEDDING_MODEL = "text-embedding-3-small"


# Reused from Lab 2, but opening a file path instead of an uploaded file
def read_pdf(path):
    pdf_doc = fitz.open(path)
    text = ""
    for page in pdf_doc:
        text += page.get_text()
    return text


def embed(text):
    # One OpenAI embedding vector for a piece of text.
    # The embedding model tops out around 8k tokens, so a very long PDF is
    # trimmed for the vector only; the full text is still stored below.
    # The 403 retry is for OpenAI's allowed-models change still propagating:
    # for a while some requests are refused and some aren't.
    for attempt in range(6):
        try:
            response = client.embeddings.create(input=text[:28000], model=EMBEDDING_MODEL)
            return response.data[0].embedding
        except PermissionDeniedError:
            if attempt == 5:
                raise
            time.sleep(2)


def create_lab4_vectordb():
    # Build the ChromaDB collection from the PDFs. This only runs once per
    # session (see the session_state check below) so we pay for embeddings once.
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="Lab4Collection")

    pdf_files = sorted(f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf"))

    # Skip anything already in the collection (another session on the same
    # server built it, or an earlier build died partway through), so a
    # half-finished build gets completed instead of frozen.
    already_in = set(collection.get()["ids"])

    for filename in pdf_files:
        if filename in already_in:
            continue
        text = read_pdf(os.path.join(PDF_FOLDER, filename))
        collection.add(
            ids=[filename],                    # key = the filename
            documents=[text],                  # full text, sent to the LLM later
            embeddings=[embed(text)],          # the vector Chroma searches on
            metadatas=[{"filename": filename}],
        )

    return collection


# Make sure the PDFs are actually there before doing anything
pdf_files = (
    sorted(f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf"))
    if os.path.isdir(PDF_FOLDER) else []
)
if not pdf_files:
    st.error("No PDFs found in Lab4_PDFs/ - copy the 7 course PDFs in first.")
    st.stop()

# Only build the vector DB once per session. The second check is a safety net:
# if an earlier build was cut short, finish it instead of trusting a partial one.
if (
    "Lab4_VectorDB" not in st.session_state
    or st.session_state.Lab4_VectorDB.count() < len(pdf_files)
):
    with st.spinner("Building the vector database (first run only)..."):
        st.session_state.Lab4_VectorDB = create_lab4_vectordb()

collection = st.session_state.Lab4_VectorDB


# ---------- Part B: the course information chatbot ----------
MODEL = "gpt-5-mini"
BUFFER_TURNS = 2   # user/assistant pairs to remember, same as Lab 3

SYSTEM_PROMPT = (
    "You are a course information assistant for the Syracuse iSchool. "
    "Below you are given the text of the course syllabi most relevant to the "
    "user's question. Answer from that material when it applies, and be clear "
    "about it: if your answer comes from the syllabi provided, say so and name "
    "the course(s). If the syllabi do not cover the question, say that the "
    "course documents don't have that information; if you then answer from "
    "general knowledge, label it as general knowledge."
)


def retrieve(question, n=3):
    # The RAG step: embed the question, get the n closest syllabi from Chroma.
    results = collection.query(query_embeddings=[embed(question)], n_results=n)
    return results["ids"][0], results["documents"][0]


st.sidebar.caption(f"{collection.count()} course documents loaded")

if "Lab4_messages" not in st.session_state:
    st.session_state.Lab4_messages = []

# Show the conversation so far (with the sources each answer used)
for msg in st.session_state.Lab4_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Retrieved: " + ", ".join(msg["sources"]))

if prompt := st.chat_input("Ask about an iSchool course..."):
    st.session_state.Lab4_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Fetch the relevant syllabi and add their text to the system prompt
    filenames, docs = retrieve(prompt)
    context = "\n\n".join(f"=== {name} ===\n{text}" for name, text in zip(filenames, docs))
    system = SYSTEM_PROMPT + "\n\nCourse documents:\n\n" + context

    buffer = st.session_state.Lab4_messages[-(BUFFER_TURNS * 2):]
    payload = [{"role": "system", "content": system}] + buffer

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=MODEL,
            messages=payload,
            stream=True,
        )
        response = st.write_stream(stream)
        st.caption("Retrieved: " + ", ".join(filenames))

    st.session_state.Lab4_messages.append(
        {"role": "assistant", "content": response, "sources": filenames}
    )
