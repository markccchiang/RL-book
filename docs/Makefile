# Minimal Sphinx makefile for the notes under ../notes
SPHINXBUILD ?= ../.venv/bin/sphinx-build
SOURCEDIR   = .
BUILDDIR    = _build

.PHONY: help html clean linkcheck
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"
	@echo "\nBuilt: docs/$(BUILDDIR)/html/index.html"

clean:
	rm -rf "$(BUILDDIR)"

linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck"
