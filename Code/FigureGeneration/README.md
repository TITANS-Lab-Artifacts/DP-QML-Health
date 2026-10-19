# Quantitative figure generation

Install `requirements.txt`, then run one script per figure from the project
root. Each script loads project results from `Results/` and writes PDF and PNG
files to `Results/Figures/Generated/`.

```bash
python Code/FigureGeneration/plot_privacy_utility.py
python Code/FigureGeneration/plot_clipping.py
python Code/FigureGeneration/plot_ablation.py
python Code/FigureGeneration/plot_ibm_validation.py
```

`generate_all_figures.py` runs those four scripts with default paths.
`generate_summaries.py` refreshes `Results/Figures/Data/`. `figure_utils.py`
contains common visual styling and plot implementations, while `io_utils.py`
contains canonical paths and input validation.

There are no background, system-overview, circuit, Draw.io, or other diagram
generators in this folder.
