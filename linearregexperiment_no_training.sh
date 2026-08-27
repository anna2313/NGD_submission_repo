#!/bin/bash

# ==========================================
# 1. Environment Setup
# ==========================================
echo "Setting up environment..."
# Source conda to ensure 'conda activate' works inside the script
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || source ~/.bashrc
conda activate ngd

# Create log directory if it doesn't exist
mkdir -p linearregexperiment/logs

# ==========================================
# 2. Initialization Step
# ==========================================
echo "Starting Initialization..."
python linearregexperiment/initialization.py --regenerate_data --sigma 0.5

# Check if initialization succeeded before continuing
if [ $? -ne 0 ]; then
    echo "Error: Initialization failed. Exiting script."
    exit 1
fi
echo "Initialization complete."

# ==========================================
# 3. Training Experiments
# ==========================================
echo "Starting Training Experiments..."

# Define hyperparameter grids based on your original script
BETA2_VALS=(0.9 0.99 0.999 0.9999)
BATCH_SIZES=(1 10 50 100 150 300)
OPTIMIZERS=("EFAdam" "ReAdam")
SEEDS=(0 1 2 3 4)

# Loop through all combinations
for beta2 in "${BETA2_VALS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        for opt in "${OPTIMIZERS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                
                # Construct the output filename to match your original pattern
                json_name="${opt}_${beta2}_${bs}_${seed}.json"
                
                echo "Running -> Opt: $opt | Beta2: $beta2 | Batch Size: $bs | Seed: $seed"
                
                # Execute the python training script
                python linearregexperiment/trainer.py \
                    --optimizer_name "$opt" \
                    --beta1 0.9 \
                    --beta2 "$beta2" \
                    --batch_size "$bs" \
                    --epochs 300 \
                    --lr 0.0 \
                    --theta 1.0 2.0 \
                    --sigma 0.5 \
                    --name_of_json_output_file "$json_name" \
                    --torch_seed "$seed" \
                    --np_seed "$seed" \
                    --random_seed "$seed" \
                    --initialization_type normal \
                    --fisher_save_period 10
                    
                # Note: These run sequentially. If one fails, the script will just move to the next.
            done
        done
    done
done

echo "All experiments finished successfully!"
