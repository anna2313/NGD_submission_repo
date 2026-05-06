#!/bin/bash

# ==========================================
# 1. Environment Setup
# ==========================================
echo "Setting up environment..."
# Source conda to ensure 'conda activate' works inside the script
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || source ~/.bashrc
conda activate ngd

# Create log directory if it doesn't exist
mkdir -p easy_classification/logs

# ==========================================
# 2. Initialization Step
# ==========================================
echo "Starting Initialization..."
python easy_classification/initialization.py

# Check if initialization succeeded before continuing
if [ $? -ne 0 ]; then
    echo "Error: Initialization failed. Exiting script."
    exit 1
fi
echo "Initialization complete."

# ==========================================
# 3. Training Experiments (Hyperparameter Sweep)
# ==========================================
echo "Starting Training Experiments..."

# Define hyperparameter grids based on your script
BETA2_VALS=(0.9 0.99 0.999 0.9999)
BATCH_SIZES=(1 10 50 100 150 300)
OPTIMIZERS=("EFAdam" "ReAdam")
LEARNING_RATES=(0.5 0.1 0.05 0.01)

# This sweep uses a constant seed of 0
SEED=0

# Loop through all combinations
for beta2 in "${BETA2_VALS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        for opt in "${OPTIMIZERS[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
                
                # Construct the output filename to match your original pattern
                # Pattern: {Optimizer}_{beta2}_{batch_size}_{seed}_{lr}.json
                json_name="${opt}_${beta2}_${bs}_${SEED}_${lr}.json"
                
                echo "Running -> Opt: $opt | Beta2: $beta2 | Batch Size: $bs | LR: $lr"
                
                # Execute the python training script
                python easy_classification/trainer.py \
                    --optimizer_name "$opt" \
                    --beta1 0.9 \
                    --beta2 "$beta2" \
                    --batch_size "$bs" \
                    --epochs 300 \
                    --lr "$lr" \
                    --theta 1.0 2.0 \
                    --sigma 0.01 \
                    --name_of_json_output_file "$json_name" \
                    --torch_seed "$SEED" \
                    --np_seed "$SEED" \
                    --random_seed "$SEED" \
                    --initialization_type normal \
                    --hyperparameter_sweep
                    
            done
        done
    done
done

echo "All easy_classification HP sweep experiments finished successfully!"
