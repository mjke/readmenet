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


DIR_DATA = '/app/data'
DIR_DATA_DOCS = f'{DIR_DATA}/docs'
DIR_CHROMA_DB = f'{DIR_DATA}/chroma'

_LANGCHAIN_COLLECTION = 'langchain'
_CHUNK_SIZE = 4000
_CHUNK_OVERLAP = 0

qa = None


def get_chromadb() -> Chroma:
    return Chroma(embedding_function = OpenAIEmbeddings(), persist_directory = DIR_CHROMA_DB)


def get_vectorstore_sources() -> list:
    docs = get_chromadb().get(include = ["metadatas"])
    return list(set([s['source'] for s in docs['metadatas']]))


def del_vectorstore_docs(sources:str):
    if len(sources) > 0:
        db = get_chromadb()
        collection = db._client.get_collection(name = _LANGCHAIN_COLLECTION,
                                               embedding_function = OpenAIEmbeddings())
        for s in sources:
            collection.delete(where = { "source" : s })
        db.persist()
    return gr.update(choices = get_vectorstore_sources())


def safe_load_vectorstore(docs_raw:list) -> Chroma:
    # check for docs already loaded (via metadata source in chromadb collection)
    existing = get_vectorstore_sources()
    docs = [d for d in docs_raw if d.metadata['source'] not in existing]

    # chunk docs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = _CHUNK_SIZE, chunk_overlap = _CHUNK_OVERLAP, separators = [" ", ",", "\n"]
    )
    texts = text_splitter.split_documents(docs)

    if len(docs) > 0:
        # calculate embeddings (most expensive step)
        db = Chroma.from_documents(texts,
                                   embedding = OpenAIEmbeddings(), # XX uses `embedding` (not _function)
                                   persist_directory = DIR_CHROMA_DB)
        db.persist()
    else:
        db = Chroma(embedding_function = OpenAIEmbeddings(), persist_directory = DIR_CHROMA_DB)
    return db, len(docs), len(docs) + len(existing)


def load_qa(db = None):
    global qa
    if db is None: db = get_chromadb()
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    qa = RetrievalQA.from_chain_type(llm = llm, chain_type = "stuff", retriever = db.as_retriever())


def load_data_directory(data_path:str = DIR_DATA) -> list:
    # load docs in datadir
    loader = DirectoryLoader(data_path, glob="*.docx", 
                            use_multithreading=True,
                            loader_cls=UnstructuredWordDocumentLoader)
    db, n_newdocs, n_sources = safe_load_vectorstore(loader.load())
    load_qa(db)
    return f"Loaded {n_newdocs} new. Vector DB contains {n_sources} total.", \
        gr.update(choices = get_vectorstore_sources())


def run_query(query:str, chat_history:list) -> list:
    if qa is None: load_qa()
    response = qa.run(query)
    chat_history.append((query, response))
    return "", chat_history


with gr.Blocks() as demo:
    gr.Markdown("LLM ADVISOR")
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot()
        text_query = gr.Textbox(label="Query", lines=6, placeholder="Query here...")
        button_clear = gr.Button("Clear")
    with gr.Tab("Load docs"):
        data_dir = gr.Textbox(label="Data directory", value=DIR_DATA_DOCS)
        text_outcome = gr.Textbox(label="Outcome")
        button_load = gr.Button("Process data")
    with gr.Tab('Manage docs'):
        boxes_sources = gr.CheckboxGroup(get_vectorstore_sources(), label="Docs", info="Stored in vectordb")
        button_delete = gr.Button("Delete selected docs")

    # chatbot
    text_query.submit(run_query, inputs = [text_query, chatbot], outputs = [text_query, chatbot])
    button_clear.click(lambda: None, None, chatbot, queue = False)

    # config
    button_load.click(load_data_directory, inputs=data_dir, outputs=[text_outcome, boxes_sources])
    button_delete.click(del_vectorstore_docs, inputs=boxes_sources, outputs=boxes_sources)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")