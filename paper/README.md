# GeoDiff-GAN Paper

`main.tex` contains the first seven pages of the planned ten-page paper:

1. abstract and introduction;
2. background and problem formulation;
3. data construction;
4. architecture;
5. evidence control and adversarial learning;
6. training and prompt conditioning;
7. experimental protocol and references.

Pages 8-10 are intentionally reserved for results, discussion, failures, and the final
conclusion after complete experiment outputs are available.

The self-contained `figures/` directory includes ten image-led explanatory infographics generated
with the built-in image-generation workflow, plus a code-derived layer-level architecture poster
in PNG and SVG formats. The explanatory images use satellite scenes, layered residuals,
feature-map cutaways, spatial masks, magnifying instruments, and uncertainty heatmaps. The
technical poster reports exact tensor dimensions, submodules, skip paths, conditioning routes,
training-only critics, and SR/edit output control. Generation details are recorded in
`INFOGRAPHICS.md`.

Regenerate the exact architecture poster after changing model dimensions:

```bash
python build_architecture_infographic.py
```

Compile from this directory with a standard LaTeX distribution:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The source uses explicit page boundaries to keep the current manuscript organized as seven
method/design pages. Check the compiled page count after changing fonts, margins, captions, or
figure sizes.
