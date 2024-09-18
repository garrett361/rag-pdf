QUERY = "What is the name of the project? Please respond in JSON format."
HOSTED_CHAT = "http://llama-31-70b-jordan.models.mlds-kserve.us.rdlabs.hpecorp.net/v1"
HOSTED_EMBED = "http://embedding-tyler.models.mlds-kserve.us.rdlabs.hpecorp.net/v1"

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
	python -m rag.parse --input private/RFQ_Commercial/NZT --output private/test/parsed --chunking_strategy "by_title" --folder_tags --combine_text_under_n_chars 50 --max_characters 1500 --new_after_n_chars 1500

.PHONY: test-embed
test-embed:
	python -m rag.embed --data-path private/test/parsed --path-to-db private/test/embedded


.PHONY: test-embed-hosted
test-embed-hosted:
	python -m rag.embed --data-path private/test/parsed --path-to-db private/test/embedded --embedding_model_path ${HOSTED_EMBED}

.PHONY: test-query
test-query:
	python -m rag.query --query '${QUERY}' --path-to-db private/test/embedded --model-name meta-llama/Meta-Llama-3.1-8B-Instruct --top-k-retriever 5

.PHONY: test-query-hosted
test-query-hosted:
	python -m rag.query --query '${QUERY}' --path-to-db private/test/embedded --model-name meta-llama/Meta-Llama-3.1-70B-Instruct --top-k-retriever 5 --chat-model-endpoint ${HOSTED_CHAT} --embedding_model_path ${HOSTED_EMBED}


.PHONY: test-query-file-hosted
test-query-file-hosted:
	python -m rag.query --query-file test_queries.txt --path-to-db private/test/embedded --model-name meta-llama/Meta-Llama-3.1-70B-Instruct --top-k-retriever 5 --chat-model-endpoint ${HOSTED_CHAT} --embedding_model_path ${HOSTED_EMBED}

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
