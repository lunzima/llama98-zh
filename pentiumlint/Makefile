# pentiumlint / cyrixlint - pre-P6 / P4 x86 inefficiency linter.
#
#   make test      run the unit + integration tests
#   make lint DIS="path/*.dis"   scan one or more disassembly files

PYTHON ?= python3
DIS ?=

.PHONY: test lint

test:
	$(PYTHON) test_pentiumlint.py

lint:
	@test -n "$(DIS)" || { echo "usage: make lint DIS='path/*.dis'"; exit 2; }
	$(PYTHON) pentiumlint.py $(DIS)
