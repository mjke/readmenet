#!/usr/bin/env python

# 2023/05 mjke
# can also develop in jupyter notebooks quickly:
# https://gradio.app/developing-faster-with-reload-mode/#jupyter-notebook-magic
# docs: https://gradio.app/docs

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader

import gradio as gr

DIR_DATA = 'data/' # assumes called from parent dir (workdir)
DIR_CHROMA_DB = f'{DIR_DATA}/chroma'

qa = None

def load_data_directory(data_path:str = DIR_DATA) -> str:
    # load docs in datadir
    loader = DirectoryLoader(data_path, glob="*.docx", 
                            use_multithreading=True, 
                            #show_progress=True,
                            loader_cls=UnstructuredWordDocumentLoader)
    docs = loader.load()

    # chunk docs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, chunk_overlap=0, separators=[" ", ",", "\n"]
    )
    texts = text_splitter.split_documents(docs)

    # calculate embeddings (most expensive step)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings, persist_directory=DIR_CHROMA_DB)
    db.persist()

    # prepare model
    global qa
    retriever = db.as_retriever()
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return f"Loaded OK? {qa is not None}"


def run_query(query:str) -> str:
    if qa is None:
        return "Need to load data first!"
    return qa.run(query)


with gr.Blocks() as demo:
    gr.Markdown("LLM ADVISOR")
    with gr.Tab("Config"):
        data_dir = gr.Textbox(label="Data directory", value=DIR_DATA)
        text_outcome = gr.Textbox(label="Outcome")
        button_data = gr.Button("Load data")
    with gr.Tab("Run"):
        text_query = gr.Textbox(label="Query", lines=6, placeholder="Query here...")
        text_output = gr.Textbox(label="Response", lines=12, placeholder="")
        button_query = gr.Button("Ask")

    button_data.click(load_data_directory, inputs=data_dir, outputs=text_outcome)
    button_query.click(run_query, inputs=text_query, outputs=text_output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")