.PHONY: help docs install format lint test

SHELL=/bin/bash

help: ## Shows this help message
	# $(MAKEFILE_LIST) is set by make itself; the following parses the `target:  ## help line` format and adds color highlighting
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'

format:  ## Formats code with `autoflake`, `black` and `isort`
	autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place pems_regression scripts --exclude=__init__.py
	black pems_regression scripts
	isort pems_regression scripts

lint:  ## Lints code with `flake8`, `black` and `isort`
	black pems_regression scripts --check --diff
	isort pems_regression scripts --check-only --diff
	flake8 pems_regression scripts