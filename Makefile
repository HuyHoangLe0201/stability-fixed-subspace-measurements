PAPER_DIR := paper

.PHONY: all pdf clean figures

all: pdf

pdf: $(PAPER_DIR)/main.pdf

$(PAPER_DIR)/main.pdf: $(wildcard $(PAPER_DIR)/*.tex) $(wildcard $(PAPER_DIR)/figures/*.pdf)
	cd $(PAPER_DIR) && TEXINPUTS=../tex//: pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	cd $(PAPER_DIR) && TEXINPUTS=../tex//: pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null

figures:
	python scripts/reproduce.py
	python scripts/reproduce_core.py
	python scripts/reproduce_rank.py

clean:
	rm -f $(PAPER_DIR)/main.aux $(PAPER_DIR)/main.log $(PAPER_DIR)/main.out $(PAPER_DIR)/main.toc
