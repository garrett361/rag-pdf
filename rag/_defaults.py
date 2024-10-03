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
DEFAULT_CUTOFF = None
DEFAULT_TOP_K_RETRIEVER = 30
DEFAULT_COMBINE_TEXT_UNDER_N_CHARS = 0
DEFAULT_MAX_CHARACTERS = 1500
DEFAULT_NEW_AFTER_N_CHARS = 500
INFORMATIVE_PROMPT = dedent("""
I will ask you if the text extract below looks informative. Examples of uninformative extracts include headers, footers, and random gibberish characters.

---------------------
{context}
---------------------

Was the above extract informative? Only respond with "yes" or "no".
""").strip("\n")

QA_PROMPT = dedent("""
Please write a list of three or fewer questions which can be aswered from the information provided in the document extract below?

---------------------
{context}
---------------------

Write the list of three or fewer questions, and nothing else, below.  Questions:\n
""")


DEFAULT_SCORE_PROMPT = dedent("""
You are an assistant who helps determine how relevant a given text excerpt is for answering a query. You will be provided the excerpt followed by the query.

Rank the relevance of the excerpt for the query on a scale of 1 to 10. 1 means the excerpt is irrelevant, while 10 means it is very relevant. Respond only with this digit and nothing else.
""").strip("\n")
