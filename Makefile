install:
	pip install -e ".[dev]"

install-uv:
	uv pip install -e ".[dev]"

lint:
	pre-commit run --all-files

test:
	pytest tests/
