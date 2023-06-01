SRC_DIR := src
SRC_EXT := py
LOG_DIR := logs
REQ_FILE := requirements-gpu.txt

# Set the requirements file to use if GPU is available
ifeq ($(shell which nvcc),)
	REQ_FILE := requirements-cpu.txt
else
	REQ_FILE := requirements-gpu.txt
endif

.PHONY: env
env:
	python3 -m pip install -U pip setuptools wheel
	echo "Installing requirements from $(REQ_FILE)"
	python3 -m pip install -r $(REQ_FILE)

.PHONY: clean
clean:
	rm -rf $(LOG_DIR)/*
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

.PHONY: format
lint:
	flake8 $(SRC_DIR) --ignore=F841,W503

.PHONY: format
format:
	isort $(SRC_DIR)
	black --line-length 79 $(SRC_DIR)
