import os
import asyncio

import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

from google.generativeai.types import (
    BlockedPromptException,
    StopCandidateException,
    BrokenResponseError,
    IncompleteIterationError,
)


load_dotenv()
# Fix event loop for Streamlit
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# read all pdf files and return text


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# split text into chunks


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk


def get_vector_store(chunks):
    if not chunks:
        st.error("No text chunks to process. The PDF might be empty or unreadable.")
        return False
    try:
        print("Chunks:", len(chunks))

        embeddings = HuggingFaceEmbeddings(
           model_name="sentence-transformers/all-MiniLM-L6-v2",
           model_kwargs={"device": "cpu"}
        )

        print("Embeddings created")

        vector_store = FAISS.from_texts(
            chunks,
            embedding=embeddings
        )

        print("FAISS created")

        vector_store.save_local("faiss_index")

        print("FAISS saved")

        return True
    except BlockedPromptException as e:
        st.error("The PDF content was flagged by Google's safety filters. Please try a different document.")
        print(f"Embedding blocked: {e}")
        return False
    except Exception as e:
        import traceback
        traceback.print_exc()
        st.error(f"Error processing the PDF: {str(e)}")
        print(f"Embedding error: {e}")
        return False


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context.

    If the answer is not available in the context,
    say only:
    "answer is not available in the context"

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=prompt
    )

    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

def user_input(user_question):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        docs = db.similarity_search(user_question)

        chain = get_conversational_chain()

        response = chain.invoke({
            "input_documents": docs,
            "question": user_question
        })

        print(response)

        return {
            "output_text": response["output_text"]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()

        return {
            "output_text": f"Error: {str(e)}"
        }


def main():
    st.set_page_config(
        page_title="Gemini PDF Chatbot",
        page_icon="🤖"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    if get_vector_store(text_chunks):
                        st.success("Done")
            else:
                st.error("Please upload at least one PDF file before processing.")

    # Main content area for displaying chat messages
    st.title("Chat with PDF files using Gemini🤖")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get bot response for the user's question
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                if response and 'output_text' in response:
                    output_text = response['output_text']
                    # Handle both string responses (from error handling) and iterable responses
                    if isinstance(output_text, str):
                        full_response = output_text
                        placeholder.markdown(full_response)
                    else:
                        for item in output_text:
                            full_response += item
                            placeholder.markdown(full_response)
                    message = {"role": "assistant", "content": full_response}
                    st.session_state.messages.append(message)
                else:
                    st.error("Failed to get a valid response. Please try again.")


if __name__ == "__main__":
    main()