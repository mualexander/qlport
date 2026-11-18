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

# Two builds from one source: main.tex defers its apparatus to appendices, main_full.tex the
# complete version with every appendix inline. The prose exists once and only its
# placement differs, so the two cannot disagree.
PAPERS := main main_full

paper:
	@command -v pdflatex >/dev/null || { echo "pdflatex not found: sudo apt install texlive-latex-extra texlive-fonts-recommended"; exit 1; }
	@for m in $(PAPERS); do \
	  (cd paper && pdflatex -interaction=nonstopmode -halt-on-error $$m.tex >/dev/null \
	    && pdflatex -interaction=nonstopmode -halt-on-error $$m.tex >/dev/null) \
	    || { echo "paper/$$m.tex FAILED"; exit 1; }; \
	  echo "paper/$$m.pdf built: $$(grep -oE '[0-9]+ pages' paper/$$m.log | tail -1)"; \
	done

all: test notebooks paper

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc
	rm -f notebooks/*.ipynb

distclean: clean
	rm -rf $(VENV)
