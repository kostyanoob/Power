# This script reproduces the plots presented in the article.
# The script trains all the neural networks and evaluates
# them against the persistent guess and the WLS estimation.
# Once the script finishes, the produced plots can be found at:
# "Figures/Article_figures"
python model_run_script.py -model-name dsse -model-type wls -dataset-name ieee37_smooth_ord_60_downsampling_factor_60 -seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,25,26,27,28,29,30 -gpuid -1 -logdir Logs/Article_run_logs -Nvs 0 -Nss 35,28,18,12,6 -Ts 5,50 -lambdas 0 --run-only-if-log-incomplete
python model_run_script.py -model-name dsse -model-type neuralnet -dataset-name ieee37_smooth_ord_60_downsampling_factor_60 -Ts 5 -lambdas 0.0 -seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,25,26,27,28,29,30 -Nss 6,12,18,28,35 --run-only-if-log-incomplete -gpuid 0  -logdir Logs/Article_run_logs &
python model_run_script.py -model-name dsse -model-type neuralnet -dataset-name ieee37_smooth_ord_60_downsampling_factor_60 -Ts 5 -lambdas 1.0 -seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,25,26,27,28,29,30 -Nss 6,12,18,28,35 --run-only-if-log-incomplete -gpuid 1  -logdir Logs/Article_run_logs &
python model_run_script.py -model-name dsse -model-type neuralnet -dataset-name ieee37_smooth_ord_60_downsampling_factor_60 -Ts 5 -lambdas 2.0 -seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,25,26,27,28,29,30 -Nss 6,12,18,28,35 --run-only-if-log-incomplete -gpuid 2  -logdir Logs/Article_run_logs &
python model_run_script.py -model-name dsse -model-type neuralnet -dataset-name ieee37_smooth_ord_60_downsampling_factor_60 -Ts 5 -lambdas 20.0 -seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,25,26,27,28,29,30 -Nss 6,12,18,28,35 --run-only-if-log-incomplete -gpuid 3  -logdir Logs/Article_run_logs &
python model_run_script.py -model-name dsse -model-type neuralnet -dataset-name ieee37_smooth_ord_60_downsampling_factor_60 -Ts 50 -lambdas 0.0 -seeds 1,2,3,4,5,6,7,8,9,10 -Nss 6,12,18,28,35 --run-only-if-log-incomplete -gpuid 0  -logdir Logs/Article_run_logs &
python model_run_script.py -model-name dsse -model-type neuralnet -dataset-name ieee37_smooth_ord_60_downsampling_factor_60 -Ts 50 -lambdas 1.0 -seeds 1,2,3,4,5,6,7,8,9,10 -Nss 6,12,18,28,35 --run-only-if-log-incomplete -gpuid 1   -logdir Logs/Article_run_logs &
python model_run_script.py -model-name dsse -model-type neuralnet -dataset-name ieee37_smooth_ord_60_downsampling_factor_60 -Ts 50 -lambdas 2.0 -seeds 1,2,3,4,5,6,7,8,9,10 -Nss 6,12,18,28,35 --run-only-if-log-incomplete -gpuid 2   -logdir Logs/Article_run_logs &
python model_run_script.py -model-name dsse -model-type neuralnet -dataset-name ieee37_smooth_ord_60_downsampling_factor_60 -Ts 50 -lambdas 20.0 -seeds 1,2,3,4,5,6,7,8,9,10 -Nss 6,12,18,28,35 --run-only-if-log-incomplete -gpuid 3   -logdir Logs/Article_run_logs &
wait
python dsse.py -model-name persistent_ieee37_smooth_ord_60_downsampling_factor_60 -model-type persistent -dataset-name ieee37_smooth_ord_60_downsampling_factor_60 --no-prediction-plots -logdir Logs/Article_run_logs
python generate_plots.py -seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 -Ts 5 -Nss 6,12,18,28,35 -lambdas 0.0,1.0,2.0,20.0 -input-data-folder Logs/Article_run_logs -output-directory Figures/Article_figures/Compare_T5_NN_lambdas -model-type-list neuralnet -non-parametric-model-name-list {}
python generate_plots.py -seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 -Ts 5 -Nss 35,28,18,12,6 -lambdas 2.0 -input-data-folder Logs/Article_run_logs -output-directory Figures/Article_figures/Compare_T5_NN_WLS_Persistent -model-type-list neuralnet,wls
python generate_plots.py -seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 -Ts 5,50 -Nss 6,12,18,28,35 -lambdas 2.0 -input-data-folder Logs/Article_run_logs -output-directory Figures/Article_figures/Compare_T5_T50 -model-type-list neuralnet,wls  -legend-x 0.1 -legend-y 0.37
