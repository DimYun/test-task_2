.PHONY:*

SYSTEM_PYTHON=~/.pyenv/versions/3.9.17/bin/python
VENV=/opt/python_venvs/test_EI-group
PYTHON=$(VENV)/bin/python3
PIP=$(VENV)/bin/pip

install:
	$(PIP) install -r requirements.txt

train:
	PYTHONPATH=. $(PYTHON) train.py configs/config.yaml

lint:
	PYTHONPATH=. $(VENV)/bin/black train.py src
	PYTHONPATH=. $(VENV)/bin/nbstripout notebooks/*.ipynb
	PYTHONPATH=. $(VENV)/bin/tox