install:
	pip install -e ".[dev]"

lint:
	pre-commit run --all-files

test:
	pytest tests/
