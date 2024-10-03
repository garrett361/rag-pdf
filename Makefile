# Pass ARGS for being able to pass additional args via the command line: make xxx ARGS="<args-here>"

QUERY = "What is the plant composition?"
HOSTED_CHAT = "http://llama-31-70b-jordan.models.mlds-kserve.us.rdlabs.hpecorp.net/v1"
# HOSTED_CHAT = "http://llama-3-1-8b.pdk.10.6.39.90.sslip.io/v1"
HOSTED_EMBED = "http://embedding-tyler.models.mlds-kserve.us.rdlabs.hpecorp.net/v1"
# HOSTED_EMBED = "http://embedding-model.pdk.10.6.39.90.sslip.io/v1"
INPUT_DIR = "private/RFQ_Commercial/"
FOLDER = "Petrobras"
OUTPUT_FOLDER = "private/test/query"
PATH_TO_DB = "private/test/embedded"
MODEL_NAME_LOCAL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME_HOSTED = "meta-llama/Meta-Llama-3.1-70B-Instruct"
# MODEL_NAME_HOSTED = "meta-llama/Meta-Llama-3.1-8B-Instruct"
QUERY_FILE = "test_queries.txt"
TOP_K_RERANKER = "2"

.PHONY: install
install:
	pip install -e .

.PHONY: fmt
fmt:
	ruff format rag

.PHONY: check
check:
	ruff check rag

.PHONY: clean
clean:
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf *.egg-info/
	rm -rf pip-wheel-metadata/
	rm -rf build/
	find . \( -name __pycache__ -o -name \*.pyc \) -delete
	rm -rf private/test/*

.PHONY: test-parse
test-parse:
	python -m rag.parse --input $(INPUT_DIR) --output private/test/parsed --folder_tags $(ARGS)

.PHONY: test-parse-hosted-cleaned
test-parse-hosted-cleaned:
	python -m rag.parse --input $(INPUT_DIR) --output private/test/parsed --folder_tags --filter-parsed-with-llm --add-questions --model-name $(MODEL_NAME_HOSTED) --chat-model-endpoint $(HOSTED_CHAT) $(ARGS)

.PHONY: test-embed
test-embed:
	python -m rag.embed --data-path private/test/parsed --path-to-db $(PATH_TO_DB) $(ARGS)


.PHONY: test-embed-hosted
test-embed-hosted:
	python -m rag.embed --data-path private/test/parsed --path-to-db $(PATH_TO_DB) --embedding_model_path $(HOSTED_EMBED) $(ARGS)

.PHONY: test-query
test-query:
	python -m rag.query --query '$(QUERY)' --path-to-db $(PATH_TO_DB) --model-name $(MODEL_NAME_LOCAL) --folder $(FOLDER) $(ARGS)

.PHONY: test-query-hosted
test-query-hosted:
	python -m rag.query --query '$(QUERY)' --path-to-db $(PATH_TO_DB) --model-name $(MODEL_NAME_HOSTED)  --chat-model-endpoint $(HOSTED_CHAT) --embedding_model_path $(HOSTED_EMBED) --folder $(FOLDER) $(ARGS)

.PHONY: test-query-hosted-reranker
test-query-hosted-reranker:
	python -m rag.query --query '$(QUERY)' --path-to-db $(PATH_TO_DB) --model-name $(MODEL_NAME_HOSTED)  --chat-model-endpoint $(HOSTED_CHAT) --embedding_model_path $(HOSTED_EMBED) --folder $(FOLDER) --top-k-reranker $(TOP_K_RERANKER) --retrieve-with-questions $(ARGS)

.PHONY: test-query-file-hosted
test-query-file-hosted:
	python -m rag.query --query-file $(QUERY_FILE) --path-to-db $(PATH_TO_DB) --model-name $(MODEL_NAME_HOSTED) --chat-model-endpoint $(HOSTED_CHAT) --embedding_model_path $(HOSTED_EMBED) --folder $(FOLDER) --output-folder $(OUTPUT_FOLDER) $(ARGS)

.PHONY: test-query-file-hosted-reranker
test-query-file-hosted-reranker:
	python -m rag.query --query-file $(QUERY_FILE) --path-to-db $(PATH_TO_DB) --model-name $(MODEL_NAME_HOSTED) --chat-model-endpoint $(HOSTED_CHAT) --embedding_model_path $(HOSTED_EMBED) --folder $(FOLDER) --output-folder $(OUTPUT_FOLDER) --top-k-reranker $(TOP_K_RERANKER) --retrieve-with-questions $(ARGS)

.PHONY: test
test:
	$(MAKE) clean
	$(MAKE) test-parse
	$(MAKE) test-embed
	$(MAKE) test-query

.PHONY: test-hosted
test-hosted:
	$(MAKE) clean
	$(MAKE) test-parse
	$(MAKE) test-embed-hosted
	$(MAKE) test-query-hosted

.PHONY: test-ui-hosted
test-ui-hosted:
	streamlit run rag/gui.py -- --path-to-db $(PATH_TO_DB) --model-name $(MODEL_NAME_HOSTED) --embedding_model_path $(HOSTED_EMBED) --chat-model-endpoint $(HOSTED_CHAT) --streaming $(ARGS)

.PHONY: test-ui-hosted-reranker
test-ui-hosted-reranker:
	streamlit run rag/gui.py -- --path-to-db $(PATH_TO_DB) --model-name $(MODEL_NAME_HOSTED) --embedding_model_path $(HOSTED_EMBED) --chat-model-endpoint $(HOSTED_CHAT) --streaming --top-k-reranker $(TOP_K_RERANKER) --retrieve-with-questions $(ARGS)
