# Power

![High-level block diagram of our state estimation framework](Figures/HighLevel.PNG)

Physics informed, deep-learning-based state estimation for distribution electrical grids. The study exploits physical properties of the grid connectivity:
1) Admittance matrix
2) Power Flow Equations (AC-model)

This physical information is used as a regularizer during a deep neural network training. Leading to superior estimation of voltage magnitudes and angles, when compared to a persistent guess or a Weighted-Least-Squares solution.

Full article: https://arxiv.org/pdf/1910.06401.pdf

## Installation
The installaion requires Python 3.6.8. The sources and the reuquired packages can be installed by running the following code in the terminal:
```
git clone https://github.com/kostyanoob/Power.git
cd Power
pip install -r requirements.txt
```

## Action items for this repository
1) write a runme.sh script for a reproduction of the experiments, add it to 
2) consider removing generate_plots.py, as we only use generate_plots_with_wls.py.
3) fix the PFE_check.py script. Currently it doesn't accomodate the up-to-date load_dataset method.
