#!/usr/bin/env python

# 2023/05 mjke

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader

# conversation memory
# https://python.langchain.com/en/latest/modules/chains/index_examples/chat_vector_db.html
from langchain.chains import ConversationalRetrievalChain

import gradio as gr

import sys
import logging


DIR_DATA = '/app/data'
DIR_DATA_DOCS = f'{DIR_DATA}/docs'
DIR_CHROMA_DB = f'{DIR_DATA}/chroma'

PATH_LOG = f'{DIR_DATA}/chat.log'

CHUNK_SIZE = 4000
CHUNK_OVERLAP = 0

_LANGCHAIN_COLLECTION = 'langchain'
#SEARCH_K = 10

CHAIN_TYPE = 'stuff' # https://docs.langchain.com/docs/components/chains/index_related_chains
VERBOSE = True

qa_chain = None # global variable


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    src: https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


def get_chromadb() -> Chroma:
    return Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=DIR_CHROMA_DB)


def get_vectorstore_sources() -> list:
    docs = get_chromadb().get(include=["metadatas"])
    return list(set([s['source'] for s in docs['metadatas']]))


def del_vectorstore_docs(sources:str):
    if len(sources) > 0:
        db = get_chromadb()
        collection = db._client.get_collection(
            name=_LANGCHAIN_COLLECTION,
            embedding_function=OpenAIEmbeddings())
        for s in sources:
            collection.delete(where={ "source" : s })
        db.persist()
    return gr.update(choices = get_vectorstore_sources())


def safe_load_vectorstore(docs_raw:list) -> Chroma:
    # check for docs already loaded (via metadata source in chromadb collection)
    existing = get_vectorstore_sources()
    docs = [d for d in docs_raw if d.metadata['source'] not in existing]

    # chunk docs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=[" ", ",", "\n"]
    )
    texts = text_splitter.split_documents(docs)

    if len(docs) > 0:
        # calculate embeddings ($$ expensive precalculation)
        db = Chroma.from_documents(
            texts,
            embedding=OpenAIEmbeddings(), # note! uses `embedding` (not _function)
            persist_directory=DIR_CHROMA_DB)
        db.persist()
    else:
        db = Chroma(
            embedding_function=OpenAIEmbeddings(),
            persist_directory=DIR_CHROMA_DB)
    return db, len(docs), len(docs) + len(existing)


def load_chain(db = None) -> None:
    if db is None: db = get_chromadb()
    global qa_chain

    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo")

    # TODO - better custom prompt for QA
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type=CHAIN_TYPE,
        retriever=db.as_retriever(), #search_kwargs={"k": SEARCH_K}),
        verbose=VERBOSE)
    

def load_data_directory(data_path:str = DIR_DATA) -> tuple:
    loader = DirectoryLoader(data_path, glob="*.docx", 
                            use_multithreading=True,
                            loader_cls=UnstructuredWordDocumentLoader)
    db, n_newdocs, n_sources = safe_load_vectorstore(loader.load())
    load_chain(db)
    return \
        f"Loaded {n_newdocs} new. vectorDB contains {n_sources} total.", \
        gr.update(choices = get_vectorstore_sources())


def run_query(query:str, chat_history) -> tuple:
    if qa_chain is None: load_chain()
    history = [(q,a) for q,a in chat_history]
    response = qa_chain({'question': query, 'chat_history': history})
    chat_history.append((query, response['answer']))
    return "", chat_history


with gr.Blocks() as demo:
    gr.Markdown("# LLM ADVISOR")
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
    text_query.submit(run_query, inputs=[text_query, chatbot], outputs=[text_query, chatbot])
    button_clear.click(lambda: None, None, chatbot, queue=False)

    # config
    button_load.click(load_data_directory, inputs=data_dir, outputs=[text_outcome, boxes_sources])
    button_delete.click(del_vectorstore_docs, inputs=boxes_sources, outputs=boxes_sources)


if __name__ == "__main__":
    logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            filename=PATH_LOG,
            filemode='a'
            )
    log = logging.getLogger('chatlog')
    #sys.stdout = StreamToLogger(log, logging.INFO) # raises exception in container
    sys.stderr = StreamToLogger(log, logging.ERROR)

    demo.launch(server_name="0.0.0.0")


# TODO
# # more memory stuff per https://github.com/hwchase17/langchain/issues/2303#issuecomment-1548837756
# from langchain.memory import ConversationSummaryBufferMemory
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
# # https://github.com/hwchase17/langchain/issues/2303
# # https://github.com/hwchase17/langchain/issues/2303#issuecomment-1536114140
# # https://python.langchain.com/en/latest/modules/chains/index_examples/chat_vector_db.html#conversationalretrievalchain-with-map-reduce
# # works ok (but not better than just handling chat history anyhow in gradio)
# memory = ConversationSummaryBufferMemory(
#     llm=llm,
#     output_key='answer',
#     memory_key='chat_history',
#     return_messages=True)
# (not working): VectorStoreRetrieverMemory