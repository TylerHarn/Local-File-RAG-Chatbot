import streamlit as st
from ingest import index_folder
from rag import ask_question

st.set_page_config(page_title="Local File RAG Chatbot", layout="wide")
st.title("Local File RAG Chatbot")
st.caption("Point this at a folder on your PC and ask questions about what's inside it.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False

with st.sidebar:
    st.header("Index a Folder")
    folder_path = st.text_input("Folder path", placeholder=r"C:\Users\YourName\Documents")

    if st.button("Index Folder"):
        if not folder_path.strip():
            st.warning("Enter a folder path first.")
        else:
            with st.spinner("Indexing folder, this may take a while..."):
                try:
                    index_folder(folder_path.strip())
                    st.session_state.indexed = True
                    st.success("Indexing complete. You can now ask questions.")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask about your files...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = ask_question(prompt)
                st.write(answer)
                if sources:
                    st.caption("Sources:")
                    for s in sources:
                        st.code(s, language=None)
            except Exception as e:
                answer = f"Error: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})