FILENAME := doc
TEX_SRCS := $(wildcard *.tex)
TEX_SRCS := $(wildcard */*.tex)

.PHONY: $(FILENAME).pdf
$(FILENAME).pdf :
#	pdflatex main.tex
#	bibtex main
	pdflatex main.tex
	pdflatex main.tex
	mv main.pdf $(FILENAME).pdf
	rm main.aux main.log
