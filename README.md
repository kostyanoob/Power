# Physics-Informed Deep Neural Network Method for Limited Observability State Estimation of a Distribution Power Grid

![High-level block diagram of our state estimation framework](Figures/HighLevel.PNG)

The goal of the study is to accurately estimate the state of the distribution power grid, which is defined as the voltage magnitudes and angles of all the buses in the grid. Our proposed approach exploits the following physical properties of the distribution grid in order to achieve accurate state estimation:
1) Admittance matrix
2) Power Flow Equations (AC-model)

This physical information is used as a regularizer during a deep neural network (DNN) training, leading to superior estimation accuracy of the grid state. We evaluate the performance in presence of limited grid observability, where less than half of the buses report any measurement, and with a high penetration of photovaoltaic (PV) panels. We compare our DNN model against a persistent guess and a Weighted-Least-Squares solution.

Full article: https://arxiv.org/pdf/1910.06401.pdf

## Installation
The installaion requires Python 3.6.8. The sources and the required packages can be installed by running the following code in the terminal:
```
git clone https://github.com/kostyanoob/Power.git
cd Power
pip install -r requirements.txt
```

## Quick Start
The following options are available for you to quickly evaluate state estimation on a particular scenario. Each of the options below evaluates a certain model with a certain model-name. Once the evaluation is complete, various plots will be generated in the  directory, check them out. 
To produce more elaborate visualizations, including per-example visuualization and a chronological visualization: remove the ```--no-prediction-plots``` flag, then run the evaluation and check out "Figures/Predictions_<model-name>" directory.

### Evaluate a pre-trained DNN model
We provide a pre-trained model inside the Models directory. You can evaluate it using the following command:
```
python dsse.py -model-type neuralnet -model-name neuralnet_T:5_Ns:5_lambda:2.0 --no-training --restore-session -model-name-for-loading neuralnet_T:5_Ns:5_lambda:2.0 -dataset-name ieee37_smooth_ord_60_downsampling_factor_60 -n-epochs 300 -data-transform standardize -T 5 -equations-regularizer 2.0 -seed 1 -Ns 35 -gpuid 0 --no-prediction-plots
```

### Training DNN model from scratch and evaluating it
The following command will train and then evaluate the DNN model on a scenario of Ns=35 visible nodes using time-window of T=5 time steps and a PFE reguarization coefficient (lambda) of 2.0: 
```
python dsse.py -model-type neuralnet -model-name neuralnet_T:5_Ns:5_lambda:2.0_from_scratch -dataset-name ieee37_smooth_ord_60_downsampling_factor_60 -n-epochs 300 -data-transform standardize -T 5 -equations-regularizer 2.0 -seed 1 -Ns 35 -gpuid 0 --no-prediction-plots
```
Loss and MSE curve plots will be available at the "Figures" directory upon a successful training termination.
 
### Evaluating WLS model
The following command will evaluate the WLS model on a scenario of Ns=35 visible nodes using time-window of T=5 time steps. 
```
python dsse.py -model-type wls -model-name WLS_T:5_Ns:5 -dataset-name ieee37_smooth_ord_60_downsampling_factor_60 -logdir Logs -T 5 -gpuid -1 -Ns 35 --wls-weights-discriminate-hidden --wls-with-power --no-prediction-plots
```
## Reproducing the article plots
The following command performs a full training and evaluation of the neural network models, and evaluates them against the WLS mdoels. Finally, all the performance plots that appeared in the paper will be generated into directory "Figures/Article_figures".
```
sh runme.sh
```

## Citation
If you build up on our research, please cite us:
```
@article{ostrometzky2019physics,
  title={Physics-Informed Deep Neural Network Method for Limited Observability State Estimation},
  author={Ostrometzky, Jonatan and Berestizshevsky, Konstantin and Bernstein, Andrey and Zussman, Gil},
  journal={arXiv preprint arXiv:1910.06401},
  year={2019}
}
```
