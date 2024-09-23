QUERY = "Provide the list of documents with proposals related to the system architecture."
HOSTED_CHAT = "http://llama-31-70b-jordan.models.mlds-kserve.us.rdlabs.hpecorp.net/v1"
HOSTED_EMBED = "http://embedding-tyler.models.mlds-kserve.us.rdlabs.hpecorp.net/v1"
INPUT_DIR = "private/RFQ_Commercial/"
FOLDER = "Petrobras"
OUTPUT_FOLDER = "private/test/query"
PATH_TO_DB = "private/test/embedded"
MODEL_NAME_LOCAL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME_HOSTED = "meta-llama/Meta-Llama-3.1-70B-Instruct"

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
	python -m rag.parse --input ${INPUT_DIR} --output private/test/parsed --chunking_strategy "by_title" --folder_tags --combine_text_under_n_chars 50 --max_characters 1500 --new_after_n_chars 1500

.PHONY: test-embed
test-embed:
	python -m rag.embed --data-path private/test/parsed --path-to-db ${PATH_TO_DB}


.PHONY: test-embed-hosted
test-embed-hosted:
	python -m rag.embed --data-path private/test/parsed --path-to-db ${PATH_TO_DB} --embedding_model_path ${HOSTED_EMBED}

.PHONY: test-query
test-query:
	python -m rag.query --query '${QUERY}' --path-to-db ${PATH_TO_DB} --model-name ${MODEL_NAME_LOCAL} --top-k-retriever 5 --folder ${FOLDER}

.PHONY: test-query-hosted
test-query-hosted:
	python -m rag.query --query '${QUERY}' --path-to-db ${PATH_TO_DB} --model-name ${MODEL_NAME_HOSTED} --top-k-retriever 5 --chat-model-endpoint ${HOSTED_CHAT} --embedding_model_path ${HOSTED_EMBED} --folder ${FOLDER}


.PHONY: test-query-file-hosted
test-query-file-hosted:
	python -m rag.query --query-file test_queries.txt --path-to-db ${PATH_TO_DB} --model-name ${MODEL_NAME_HOSTED} --top-k-retriever 5 --chat-model-endpoint ${HOSTED_CHAT} --embedding_model_path ${HOSTED_EMBED} --folder ${FOLDER} --output-folder ${OUTPUT_FOLDER}

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
	streamlit run rag/gui.py -- --path-to-db ${PATH_TO_DB} --model-name ${MODEL_NAME_HOSTED}  --embedding_model_path ${HOSTED_EMBED} --cutoff 0.6 --chat-model-endpoint ${HOSTED_CHAT} --streaming
