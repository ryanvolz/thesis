# latexmk default options are set in latexmkrc
LATEX_ARGS = -interaction=nonstopmode -file-line-error

.PHONY: FORCE_MAKE

all: thesis.pdf

autoupdate:
	latexmk $(LATEX_ARGS) -pvc thesis

clean:
	latexmk -c thesis
	-rm thesis.run.xml

display: thesis.pdf
	xdg-open thesis.pdf

distclean:
	latexmk -C thesis
	-rm thesis.run.xml thesis.bbl

%.pdf: %.tex FORCE_MAKE
	latexmk $(LATEX_ARGS) thesis
