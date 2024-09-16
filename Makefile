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
	python -m rag.parse --input private/RFQ_Commercial/NZT --output private/test/parsed --chunking_strategy "by_title"

.PHONY: test-chunk
test-chunk:
	python -m pdb -m rag.chunk --input private/test/parsed --output private/test/chunked

.PHONY: test-embed
test-embed:
	python -m rag.embed --data-path private/test/parsed --path-to-db private/test/embedded

.PHONY: test-query
test-query:
	# python -m rag.query "What is the name of the project?" --path-to-db private/test/embedded
	# python -m rag.query "What is the name of the project?" --path-to-db private/test/embedded --model-name meta-llama/Llama-2-7b-chat-hf
	python -m rag.query "What is the name of the project?" --path-to-db private/test/embedded --model-name meta-llama/Meta-Llama-3.1-8B-Instruct

.PHONY: test
test:
	$(MAKE) clean
	$(MAKE) parse
	$(MAKE) embed
	$(MAKE) query
