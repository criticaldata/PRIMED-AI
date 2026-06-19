install:
	pip install -e .

lint:
	pre-commit run --all-files

test:
	pytest tests/