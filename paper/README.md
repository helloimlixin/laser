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

Reported evidence: this draft uses only the CelebA-HQ run group `laser-train-powerlong-0612-celebahq-p2s2k4-q256-20260612_001034` and the VCTK waveform run group `laser-train-vctkgan-0613_0705-noMel-vctk-power-ds256-p2s2k4-q256-20260613_070519`.
