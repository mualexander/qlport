VENV    := .venv
PY      := $(VENV)/bin/python
KERNEL  := qlport
NOTEBOOKS := $(sort $(wildcard notebooks/[0-9]*.py))

.PHONY: install test notebooks paper all clean distclean

# Ubuntu's system Python is "externally managed" (PEP 668), so everything runs in .venv.
$(VENV)/.installed: pyproject.toml
	python3 -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e ".[dev,regular]"
	$(PY) -m ipykernel install --user --name $(KERNEL) --display-name "Python (qlport)" >/dev/null
	touch $@

install: $(VENV)/.installed

test: install
	$(PY) -m pytest -q

# Notebooks are stored as jupytext .py files; convert and execute each one with the venv kernel.
notebooks: install
	@for nb in $(NOTEBOOKS); do \
	  echo "== $$nb"; \
	  $(PY) -m jupytext --to ipynb --set-kernel $(KERNEL) --execute "$$nb" >/dev/null || exit 1; \
	done

paper:
	@command -v pdflatex >/dev/null || { echo "pdflatex not found: sudo apt install texlive-latex-extra texlive-fonts-recommended"; exit 1; }
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null \
	  && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null \
	  && echo "paper/main.pdf built"

all: test notebooks paper

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc
	rm -f notebooks/*.ipynb

distclean: clean
	rm -rf $(VENV)
