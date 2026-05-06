#!/bin/bash

# ==========================================
# 1. Environment Setup
# ==========================================
echo "Setting up environment..."
# Source conda to ensure 'conda activate' works inside the script
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || source ~/.bashrc
conda activate ngd

# Create log directory if it doesn't exist
mkdir -p MNIST_experiment/logs

# ==========================================
# 2. Initialization Step
# ==========================================
echo "Starting Initialization..."
python MNIST_experiment/initialization.py

# Check if initialization succeeded before continuing
if [ $? -ne 0 ]; then
    echo "Error: Initialization failed. Exiting script."
    exit 1
fi
echo "Initialization complete."

# ==========================================
# 3. Training Experiments (Seed Sweep)
# ==========================================
echo "Starting Training Experiments..."

# Define hyperparameter grids based on the MNIST script
BETA2_VALS=(0.9 0.99 0.999)
BATCH_SIZES=(1 1000 5000 25000 50000)
OPTIMIZERS=("EFAdam" "ReAdam")
SEEDS=(0 1 2)

# Loop through all combinations
for beta2 in "${BETA2_VALS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        for opt in "${OPTIMIZERS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                
                # Construct the output filename to match your original pattern
                # Pattern: {Optimizer}_{beta2}_{batch_size}_{seed}.json
                json_name="${opt}_${beta2}_${bs}_${seed}.json"
                
                echo "Running -> Opt: $opt | Beta2: $beta2 | Batch Size: $bs | Seed: $seed"
                
                # Execute the python training script
                python MNIST_experiment/trainer.py \
                    --optimizer_name "$opt" \
                    --beta1 0.9 \
                    --beta2 "$beta2" \
                    --batch_size "$bs" \
                    --epochs 10 \
                    --lr 0.0 \
                    --name_of_json_output_file "$json_name" \
                    --torch_seed "$seed" \
                    --np_seed "$seed" \
                    --random_seed "$seed" \
                    --fisher_save_period 10
                    
            done
        done
    done
done

echo "All MNIST experiments finished successfully!"
