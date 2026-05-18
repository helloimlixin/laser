# LASER TMLR Draft

This directory contains a TMLR-style LaTeX draft for the LASER project.

Build with:

```bash
latexmk -pdf main.tex
```

Fallback if `latexmk` is unavailable:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The draft uses the official TMLR style files from `JmlrOrg/tmlr-style-file` and includes all referenced figures under `figures/`, so the `paper/` directory can be uploaded directly to Overleaf as one project folder.

Important caveat: the current audio LASER evidence is partial. The local artifacts include completed VQ-VAE VCTK stage-1/stage-2 runs and interrupted LASER VCTK stage-1 runs, but no completed LASER VCTK token cache or stage-2 checkpoint.
