PY := PYTHONPATH=. .venv/bin/python
UVICORN := PYTHONPATH=. .venv/bin/uvicorn
PRAX_API_URL ?= http://localhost:8000/api/v1

.PHONY: install db-up db-down seed run ui cli reset refresh docs check help

help:
	@echo "Targets: install db-up db-down seed run cli docs check"

install:
	uv pip install -e .

db-up:
	docker compose up -d

db-down:
	docker compose down

seed:
	$(PY) scripts/seed.py

run:
	$(UVICORN) app.main:app --host 127.0.0.1 --port 8000 --reload

# Launch the Streamlit chatbot UI (needs the API running).
ui:
	PRAX_API_URL=$(PRAX_API_URL) .venv/bin/streamlit run ui/app.py --server.port 8501

# Usage: make cli q="What vector db does Prax use?"
cli:
	$(PY) scripts/cli.py query "$(q)"

# Delete all documents/chunks (no confirmation prompt).
reset:
	$(PY) scripts/cli.py reset --yes

# Refresh = wipe existing data and re-load the sample documents.
refresh: reset seed

# Regenerate the code-derived sections of the docs.
docs:
	$(PY) scripts/gen_docs.py

# CI guard: fail if docs are out of date (no git required).
check:
	$(PY) scripts/gen_docs.py --check
