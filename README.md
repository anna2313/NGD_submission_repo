# Set-up description

When cloning this repository, just create the environment following the environment.yml. No other set-up is neccessary.

A general note: we have 4 set-ups (dataset, models, training files, etc.) in folders linearregexperiment, sinexperiment, easy_classification and MNIST_experiment. On these set-ups there are two experiments: (1) No training validation experiment, (2) Training experiment (this has a learning rate tuning phase).

# Experiments without training 

Run these from the root directory

Linear regression: ./linearregexperiment_no_training.sh
Sin regression: ./sinexperiment_no_training.sh
Binary classification: ./easy_classification_no_training.sh
MNIST classification: ./MNIST_experiment_no_training.sh


Acknowledgements: This repository uses some files of the ASDL: Automatic Second-order Differentiation Library https://github.com/kazukiosawa/asdl
