import argparse
import os
import pathlib

import streamlit as st
from llama_index.core import Settings
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.llms.openllm import OpenLLM
from transformers import AutoTokenizer

from rag._defaults import (
    DEFAULT_CUTOFF,
    DEFAULT_HF_CHAT_MODEL,
    DEFAULT_HF_EMBED_MODEL,
    DEFAULT_MAX_NEW_TOKS,
    DEFAULT_TEMP,
    DEFAULT_TOP_K_RETRIEVER,
    DEFAULT_TOP_P,
)
from rag.query import (
    create_retriever,
    get_llama3_1_instruct_str,
    get_llm_answer,
    get_local_llm,
    get_nodes,
    load_data,
)

static_path = pathlib.Path(__file__).parent.joinpath("static")
print(f"{static_path=}")

parser = argparse.ArgumentParser()
parser.add_argument("--path-to-db", type=str, default="db", help="path to chroma db")
parser.add_argument(
    "--embedding_model_path",
    type=str,
    default=DEFAULT_HF_EMBED_MODEL,
    help="local path or URL to sentence transformer model",
)
parser.add_argument(
    "--model-name",
    default=DEFAULT_HF_CHAT_MODEL,
    help="local path or URL to chat model",
)
parser.add_argument(
    "--chat-model-endpoint",
    default=None,
    type=str,
    help="HTTP path to model endpoint, if serving",
)
parser.add_argument(
    "--top-k-retriever",
    default=DEFAULT_TOP_K_RETRIEVER,
    type=int,
    help="top k for retreiver",
)
parser.add_argument(
    "--max-new-tokens",
    default=DEFAULT_MAX_NEW_TOKS,
    type=int,
    help="Max generation toks",
)
parser.add_argument(
    "--cutoff",
    default=DEFAULT_CUTOFF,
    type=float,
    help="cutoff for similarity score",
)
parser.add_argument("--streaming", help="stream responses", action="store_true")
args = parser.parse_args()

st.set_page_config(layout="wide", page_title="Retrieval Augmented Generation (RAG) Demo Q&A")

with open(static_path.joinpath("style.css")) as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)


######

# CSS for formatting top bar
st.markdown(
    """
    <style>
    .top-bar {
        background-color: #00B188;
        padding: 15px;
        color: white;
        margin-top: -82px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create top bar
st.markdown(
    """
    <div class="top-bar">
         <img src="app/static/hpe_pri_wht_rev_rgb.png" alt="HPE Logo" height="55">
    </div>
    """,
    unsafe_allow_html=True,
)

######

st.header("Retrieval Augmented Generation (RAG) Demo Q&A", divider="gray")

st.session_state.temp = DEFAULT_TEMP
st.session_state.top_p = DEFAULT_TOP_P
st.session_state.max_length = DEFAULT_MAX_NEW_TOKS
st.session_state.cutoff = args.cutoff
st.session_state.top_k_retriever = args.top_k_retriever

tokenizer = AutoTokenizer.from_pretrained(args.model_name)


@st.cache_data
def load_chat_model(
    temp=st.session_state.temp,
    max_length=st.session_state.max_length,
    top_p=st.session_state.top_p,
) -> None:
    generate_kwargs = {
        "do_sample": True,
        "temperature": temp,
        "top_p": top_p,
        "max_length": max_length,
    }
    if args.chat_model_endpoint:
        st.write(f"Using model endpoint: {args.model_name}")
        Settings.llm = OpenLLM(
            model=args.model_name,
            api_base=args.chat_model_endpoint,
            api_key="fake",
            generate_kwargs=generate_kwargs,
            max_tokens=args.max_new_tokens,
        )
    else:
        Settings.llm = get_local_llm(
            model_name=args.model_name,
            tokenizer=tokenizer,
            use_4_bit_quant=False,
            generate_kwargs=generate_kwargs,
        )
        st.write(f"Using local model: {args.model_name}")


welcome_message = "Hello, I am HPE Document chat. \n\n Please ask me any questions related to the documents listed below. If there are no documents listed, please select a tag below to filter."
col1, col2 = st.columns(2)

chat_container = col1.container(height=435, border=False)
input_container = col1.container()


with st.spinner(f"Loading {args.model_name} q&a model..."):
    llm = load_chat_model()

with st.spinner(f"Loading data and {args.embedding_model_path} embedding model..."):
    index, chunks = load_data(args.embedding_model_path, args.path_to_db)


tags = []
uploaded_files = {}
filters = None
for c in chunks:
    file = c.properties["source"]
    eltags = c.properties["tag"]
    if eltags not in tags:
        tags.append(eltags)
    if eltags not in uploaded_files:
        uploaded_files[eltags] = []
    if file not in uploaded_files[eltags]:
        uploaded_files[eltags].append(file)


def list_sources():
    col2.markdown("##### List of Sources:")
    global filters
    filter_tags = st.session_state["tags"] if "tags" in st.session_state else []
    if len(filter_tags) > 0:
        meta_filters = []
        for tag in filter_tags:
            with col2.expander(tag):
                files = uploaded_files[tag]
                for file in files:
                    st.write(file)
            # After moving to weaviate, needed to change key="Tag" to the lower-cased key="tag"
            meta_filters.append(MetadataFilter(key="tag", value=tag))
        filters = MetadataFilters(
            filters=meta_filters,
            condition="or",
        )
    else:
        for tag in uploaded_files:
            with col2.expander(tag):
                files = uploaded_files[tag]
                for file in files:
                    st.write(file)


if len(tags) > 0:
    filter_tags = col2.multiselect(
        "Select Tags to Filter on:", tags, on_change=list_sources(), key="tags"
    )
    col1.divider()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message to new chat history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": welcome_message,
            "avatar": str(static_path.joinpath("logo.jpeg")),
        }
    )

for message in st.session_state.messages:
    if "avatar" not in message:
        message["avatar"] = None
    with chat_container.chat_message(message["role"], avatar=message["avatar"]):
        st.write(message["content"])

default_instructions = "If you don't know the answer to a question, please don't share false information. \n Limit your response to 500 tokens."
brief = "just generate the answer without a lot of explanations."


def reload():
    with st.spinner(f"Loading {args.model_name} q&a model..."):
        load_chat_model(
            temp=st.session_state.temp,
            top_p=st.session_state.top_p,
            max_length=st.session_state.max_length,
        )
    global retriever
    retriever = create_retriever(
        index=index,
        cutoff=st.session_state.cutoff,
        top_k_retriever=st.session_state.top_k_retriever,
        filters=filters,
    )


def output_stream(llm_stream):
    for chunk in llm_stream:
        yield chunk.delta


with col1.expander("Settings"):
    temp = st.slider("Temperature", 0.0, 1.0, key="temp")
    top_k_retriever = st.slider("Top K (Retriever)", 1, 25, key="top_k_retriever")
    cutoff = st.slider("Cutoff", 0.0, 1.0, key="cutoff")
    instructions = st.text_area("Prompt Instructions", default_instructions)
    st.button("Update Settings", on_click=reload())

# Accept user input
if prompt := input_container.chat_input("Say something..."):
    with chat_container.chat_message("user"):
        st.write(prompt)

    nodes = get_nodes(prompt, retriever, reranker=None)
    prefix = get_llama3_1_instruct_str(prompt, nodes, tokenizer)
    print(f"Querying with prompt: {prompt}")
    nodes = get_nodes(prompt, retriever, reranker=None)
    response = get_llm_answer(Settings.llm, prefix, args.streaming)
    with chat_container.chat_message("assistant", avatar=str(static_path.joinpath("logo.jpeg"))):
        if args.streaming:
            st.write_stream(output_stream(response))
        else:
            st.write(response.text)

    project = os.getenv("PPS_PROJECT_NAME", "default")
    doc_repo = os.getenv("DOCUMENT_REPO", "documents")
    proxy_url = os.getenv("PACH_PROXY_EXTERNAL_URL_BASE", "http://localhost:30080")

    with col2:
        for n in nodes:
            title = n.metadata["Source"]
            page = n.metadata["PageNumber"]
            text = n.text
            commit = n.metadata["Commit"]
            doctag = n.metadata["Tag"]
            newtext = text.encode("unicode_escape").decode("unicode_escape")
            out_title = f"**Source:** {title}  \n **Page:** {page}  \n **Similarity Score:** {round((n.score * 100),3)}% \n"
            out_text = f"**Text:**  \n {newtext}  \n"
            title = title.replace(" ", "%20")
            if doctag:
                doctag = doctag.replace(" ", "%20")
                out_link = f"[Link to file in Commit {commit}]({proxy_url}/proxyForward/pfs/{project}/{doc_repo}/{commit}/{doctag}/{title}#page={page})\n"
            else:
                out_link = f"[Link to file in Commit {commit}]({proxy_url}/proxyForward/pfs/{project}/{doc_repo}/{commit}/{title}#page={page})\n"
            col2.markdown(out_title)
            col2.write(out_text, unsafe_allow_html=True)
            if not title.startswith("http"):
                col2.write(out_link)
            col2.divider()
