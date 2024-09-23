from textwrap import dedent

DEFAULT_HF_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_HF_CHAT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_CHUNK_STRAT = "by_title"
DEFAULT_HF_CHAT_TEMPLATE = "\n<|user|>:{}</s>\n<|assistant|>:"
DEFAULT_SYSTEM_PROMPT = dedent("""
You are an assistant for answering questions.
You are given the extracted parts of a long document and a question. Provide a succinct answer.
If you don't know the answer, just say "I do not know." Don't make up an answer.
""").strip("\n")
# The embedding model hosted on houston errors out at larger batch sizes
DEFAULT_MAX_EMBED_BSZ = 32
DEFAULT_MAX_NEW_TOKS = 500
DEFAULT_ALPHA = 0.95
UNINFORMATIVE_PROMPT = dedent("""
Below is a text extract from a document. I will ask you whether the extract looks informative.

Extract: {context}

Headers, footers, and tables of contents are examples of non-informative extracts. Does the above extract look uninformative? Only respond with "yes" or "no".
""").strip("\n")

QA_PROMPT = dedent("""
Generate the main question that is answered by the information provided in the passage below.  Ignore weird formatting or characters that look out of place.
{context}

Only respond with the question.
""").strip("\n")
