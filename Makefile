#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = Breast-Cancer-Classification---SE489
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

# Check types on module
.PHONY: type_check 
type_check:
	mypy .\breast_cancer_classification\

## Currently tests are not implemented
## Run tests
#.PHONY: test # By default, Makefile targets are "file targets" - they are used to build files from other files. Make assumes its target is a file, and this makes writing Makefiles relatively easy:
#test:
#	#python -m unittest discover -s tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	

process_data:
	python ./breast_cancer_classification/dataset.py

train_model:
	python ./breast_cancer_classification/modeling/train.py

test_model:
	python ./breast_cancer_classification/modeling/predict.py
	
run_full_model_pipeline:
	python run.py

# cProfile specific components
profile_train:
	python -m cProfile -o reports/profiling/train.prof ./breast_cancer_classification/modeling/train.py

profile_predict:
	python -m cProfile -o reports/profiling/predict.prof ./breast_cancer_classification/modeling/predict.py

profile_dataset:
	python -m cProfile -o reports/profiling/dataset.prof ./breast_cancer_classification/dataset.py

profile_run:
	python -m cProfile -o reports/profiling/run.prof ./run.py

# View profile outputs
view_train_profile:
	python profiling/read_train_cprofile.py reports/profiling/train.prof 

view_predict_profile:
	python profiling/read_predict_cprofile.py reports/profiling/predict.prof 

view_dataset_profile:
	python profiling/read_dataset_cprofile.py reports/profiling/dataset.prof 

view_run_profile:
	python profiling/read_run_cprofile.py reports/profiling/run.prof 

# Visualize profiles with snakeviz
viz_train_profile:
	snakeviz reports/profiling/train.prof

viz_predict_profile:
	snakeviz reports/profiling/predict.prof

viz_dataset_profile:
	snakeviz reports/profiling/dataset.prof

viz_run_profile:
	snakeviz reports/profiling/run.prof

# PyTorch train profiling

profile_pytorch_train:
	python -m profiling.pytorch_train

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
