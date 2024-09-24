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
DEFAULT_MAX_NEW_TOKS = 1024
DEFAULT_TOP_P = 0.9
DEFAULT_ALPHA = 0.2
DEFAULT_TEMP = 0.2
DEFAULT_CUTOFF = 0.1
DEFAULT_TOP_K_RETRIEVER = 10
DEFAULT_COMBINE_TEXT_UNDER_N_CHARS = 100
DEFAULT_MAX_CHARACTERS = 1000
DEFAULT_NEW_AFTER_N_CHARS = 1000
INFORMATIVE_PROMPT = dedent("""
I will ask you if the text extract below looks informative. Examples of uninformative extracts include headers, footers, and random gibberish characters.

---------------------
{context}
---------------------

Was the above extract informative? Only respond with "yes" or "no".
""").strip("\n")

QA_PROMPT = dedent("""
What question can be answered from the information provided in the document extract below?

---------------------
{context}
---------------------

Only respond with the question.
""").strip("\n")
