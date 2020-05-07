# Power

![High-level block diagram of our state estimation framework](Figures/HighLevel.PNG)

Physics informed, deep-learning-based state estimation for distribution electrical grids. The study exploits physical properties of the grid connectivity:
1) Admittance matrix
2) Power Flow Equations (AC-model)
This physical information is used as a regularizer during a deep neural network training. Leading to superior estimation of voltage magnitudes and angles, when compared to a persistent guess or a Weighted-Least-Squares solution.

Full article: https://arxiv.org/pdf/1910.06401.pdf
