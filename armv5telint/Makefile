# armv5telint - ARMv5TE peephole linter.
#
#   make test      run the unit tests
#   make lint DIS="path/*.dis"   scan one or more objdump -d outputs

PYTHON ?= python3
DIS ?=

.PHONY: test lint

test:
	$(PYTHON) test_armv5telint.py

lint:
	@test -n "$(DIS)" || { echo "usage: make lint DIS='path/*.dis'"; exit 2; }
	$(PYTHON) armv5telint.py $(DIS)
