# Set-up description

When cloning this repository, just create the environment following the environment.yml. No other set-up is neccessary.

A general note: we have 4 set-ups (dataset, models, training files, etc.) in folders linearregexperiment, sinexperiment, easy_classification and MNIST_experiment. On these set-ups there are two experiments: (1) No training validation experiment, (2) Training experiment (this has a learning rate tuning phase).

# Experiments without training 

Run these from the root directory

Linear regression: ./linearregexperiment_no_training.sh
Sin regression: ./sinexperiment_no_training.sh
Binary classification: ./easy_classification_no_training.sh
MNIST classification: ./MNIST_experiment_no_training.sh

then for each folder run python {folder_name}/merge_run_jsons.py --optimizer=EFAdam and {folder_name}/merge_run_jsons.py --optimizer=ReAdam
then to get the plots for each folder run 

python plot_results.py {folder_name}/results/results_ReAdam.json {folder_name}/results/results_EFAdam.json --beta2 {beta_values} --batch-size {batch_sizes} --fisher-types 'adam' --output-path {folder_name}
python plot_results.py {folder_name}/results/results_EFAdam.json --beta2 {beta_values}  --batch-size {batch_sizes} --fisher-types 'adam' 'empirical' --output-path {folder_name}
python plot_results.py {folder_name}/results/results_ReAdam.json {folder_name}/results/results_EFAdam.json --beta2 {beta_values} --batch-size {batch_sizes} --fisher-types 'empirical' --output-path {folder_name}
python plot_results.py {folder_name}/results/results_ReAdam.json --beta2 {beta_values}  --batch-size {batch_sizes} --fisher-types 'adam' 'empirical' --output-path {folder_name}

(for linear regression, sin regression and binary classification {batch_sizes} should be 1 10 50 100 150 300 and {beta_values} should be 0.9 0.99 0.999 0.9999 and for MNIST they should be 1 1000 5000 25000 50000 and 0.9 0.99 0.999 respectively)

# Experiments with training 

First we ran hyperparameter tuning on the learning rates. For this run the following sh files:

Linear regression: ./linearregexperiment_hyperparam_tuning.sh
Sin regression: ./sinexperiment_hyperparam_tuning.sh
Binary classification: ./easy_classification_hyperparam_tuning.sh
MNIST classification: ./MNIST_experiment_hyperparam_tuning.sh

Afterwards run python {folder_name}/merge_run_hyper_jsons.py --optimizer EFAdam ReAdam --output {folder_name}/saved_results

For the actual training run the following sh files:

Linear regression: ./linearregexperiment_training.sh
Sin regression: ./sinexperiment_training.sh
Binary classification: ./easy_classification_training.sh
MNIST classification: ./MNIST_experiment_training.sh


Acknowledgements: This repository uses some files of the ASDL: Automatic Second-order Differentiation Library https://github.com/kazukiosawa/asdl
