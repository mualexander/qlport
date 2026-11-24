VENV    := .venv
PY      := $(VENV)/bin/python
KERNEL  := qlport
NOTEBOOKS := $(sort $(wildcard notebooks/[0-9]*.py))

.PHONY: install test notebooks data paper all clean distclean

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

# Scripts that write what the tests read. data/processed/ is gitignored because it is
# derived, so a fresh checkout has none of it: the notebooks produce most of it from the
# corpus and these three produce the rest. Until they have run, five tests over the
# promtool comparison and four over the rollup join have no input to read, which is a
# failing suite on a clean clone and was so for three days before anyone noticed.
#
# Deliberately not a dependency of `test`. promtool_parse.py needs docker and node, and a
# machine with neither should still be able to run the suite against data it already has
# rather than failing at the first container. Order matters: both read
# data/processed/translations.csv, which notebook 03 writes, so `notebooks` comes first.
#
# None is passed --write-registry. The registry is committed with these values in it, and
# a reproduction run should agree with what is committed rather than quietly rewrite it;
# pass the flag by hand when a figure is meant to move. dash0_claims.py writes nothing at
# all -- it re-checks the cached README against its sha256 and re-reads the two figures
# section 2 attributes to that converter.
DATA_SCRIPTS := scripts/rollup_span_join.py scripts/promtool_parse.py \
                scripts/dash0_claims.py scripts/heldout_split.py \
                scripts/heldout_eval.py

data: install
	@for s in $(DATA_SCRIPTS); do \
	  echo "== $$s"; \
	  $(PY) "$$s" >/dev/null || { echo "$$s FAILED"; exit 1; }; \
	done
	@echo "data/processed is current"

# Two builds from one source: main.tex defers its apparatus to appendices, main_full.tex the
# complete version with every appendix inline. The prose exists once and only its
# placement differs, so the two cannot disagree.
#
# The other two are twenty-page versions, and they are different documents rather than
# further renderings of this one: main_short.tex carries the inexpressibility spine and
# main_switching.tex the empirical one. Both are built here so that no numeral in either
# escapes the digest or the figure census.
PAPERS := main main_full main_short main_switching

paper:
	@command -v pdflatex >/dev/null || { echo "pdflatex not found: sudo apt install texlive-latex-extra texlive-fonts-recommended"; exit 1; }
	@for m in $(PAPERS); do \
	  (cd paper && pdflatex -interaction=nonstopmode -halt-on-error $$m.tex >/dev/null \
	    && pdflatex -interaction=nonstopmode -halt-on-error $$m.tex >/dev/null \
	    && pdflatex -interaction=nonstopmode -halt-on-error $$m.tex >/dev/null) \
	    || { echo "paper/$$m.tex FAILED"; exit 1; }; \
	  echo "paper/$$m.pdf built: $$(grep -oE '[0-9]+ pages' paper/$$m.log | tail -1)"; \
	done
# What the build actually read, by content. The suite's staleness check was an mtime
# comparison, so `touch paper/main.log` satisfied every test downstream of it against a
# build that no longer existed. A digest cannot be touched into agreement.
	@$(PY) scripts/stamp_sources.py

# A fresh checkout, in the order the dependencies actually run: the corpus becomes
# data/processed, the scripts finish it, the paper quotes it, the suite reads both.
# `paper` comes before `test` because the build-log layer checks a content digest of
# what the build read, so the suite now reads the paper's build products -- and the
# old ordering made `make all` on a fresh clone fail at `test` without ever reaching
# `paper`, which is the order the comment itself used to describe.
all: install notebooks data paper test

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc paper/.sources.sha256
	rm -f notebooks/*.ipynb

distclean: clean
	rm -rf $(VENV)
