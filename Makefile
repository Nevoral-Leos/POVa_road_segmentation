ENV_DIR = env
PYTHON = python3
SRC_DIR = src/DeepGlobe
DATA_DIR = data
SCRIPT = $(SRC_DIR)/main.py
DEEP_GLOBE_DATASET_URL = https://www.kaggle.com/api/v1/datasets/download/balraj98/deepglobe-road-extraction-dataset
DEEP_GLOBE_ARCHIVE = $(DATA_DIR)/deepglobe-road-extraction-dataset.zip
MASSACHUSETTS_DATASET_URL = https://www.kaggle.com/api/v1/datasets/download/balraj98/massachusetts-roads-dataset
MASSACHUSETTS_ARCHIVE = $(DATA_DIR)/massachusetts-roads-dataset.zip

all: run

.PHONY: help clean install deepglobe massachusetts run

venv:
	$(PYTHON) -m venv $(ENV_DIR)

install: venv
	$(ENV_DIR)/bin/pip install --upgrade pip
	$(ENV_DIR)/bin/pip install -r requirements.txt

run:
	$(ENV_DIR)/bin/python $(SCRIPT)

clean:
	rm -rf $(ENV_DIR)

$(DEEP_GLOBE_ARCHIVE):
	@echo "Downloading dataset..."
	mkdir -p $(DATA_DIR)
	curl -L -o $@ $(DEEP_GLOBE_DATASET_URL)

$(MASSACHUSETTS_ARCHIVE):
	@echo "Downloading dataset..."
	mkdir -p $(DATA_DIR)
	curl -L -o $@ $(MASSACHUSETTS_DATASET_URL)

deepglobe: $(DEEP_GLOBE_ARCHIVE)
	@echo "Extracting dataset..."
	unzip -o $(DEEP_GLOBE_ARCHIVE) -d $(DATA_DIR)/deepglobe

massachusetts: $(MASSACHUSETTS_ARCHIVE)
	@echo "Extracting dataset..."
	unzip -o $(MASSACHUSETTS_ARCHIVE) -d $(DATA_DIR)/massachusetts

help:
	@echo "Makefile for managing the project"
	@echo "Targets:"
	@echo "  install       - Create virtual environment and install dependencies"
	@echo "  deepglobe     - Prepare the deepglobe dataset"
	@echo "  massachusetts - Prepare the dataset"
	@echo "  run           - Run the script in the local environment"
	@echo "  clean         - Remove the virtual environment"
