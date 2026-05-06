import json
import os

# ==========================================
# Configuration
# ==========================================
# Point this to your specific JSON file
json_file_path = "sinexperiment/saved_results/best_best_learning_rates_selected_optimizers.json"
output_sh_file = "run_best_sine_experiments_local.sh"

# Check if the JSON file exists
if not os.path.exists(json_file_path):
    print(f"Error: Could not find {json_file_path}")
    print("Please make sure the path is correct and the file exists.")
    exit(1)

# Read the JSON file
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Open the new bash script for writing
with open(output_sh_file, 'w') as out_file:
    # Write the bash header and environment setup
    out_file.write("#!/bin/bash\n\n")
    out_file.write("echo 'Setting up environment...'\n")
    out_file.write("source \"$(conda info --base)/etc/profile.d/conda.sh\" 2>/dev/null || source ~/.bashrc\n")
    out_file.write("conda activate ngd\n\n")
    out_file.write("mkdir -p sinexperiment/logs\n\n")
    
    out_file.write("echo 'Starting Initialization...'\n")
    out_file.write("python sinexperiment/initialization.py --regenerate_data --sigma 0.3\n\n")
    
    out_file.write("if [ $? -ne 0 ]; then\n")
    out_file.write("    echo 'Error: Initialization failed. Exiting script.'\n")
    out_file.write("    exit 1\n")
    out_file.write("fi\n\n")
    
    out_file.write("echo 'Initialization complete. Starting Training Experiments...'\n\n")

    # Assuming your JSON structure matches the previous one:
    # A dictionary containing a "best_configs_above_threshold" list
    configs = data.get("best_configs_above_threshold", [])
    
    if not configs:
         print(f"Warning: No 'best_configs_above_threshold' found in {json_file_path}.")

    # Loop through the best configurations from the JSON
    for config in configs:
        opt = config.get("optimizer")
        beta2 = config.get("beta2")
        bs = config.get("batch_size")
        lr = config.get("best_learning_rate")
        
        # Skip if any essential data is missing
        if any(v is None for v in [opt, beta2, bs, lr]):
            continue
            
        out_file.write(f"# === {opt} | Beta2: {beta2} | Batch Size: {bs} | Best LR: {lr} ===\n")
        
        # Generate a run for seeds 0 through 4
        for seed in range(5):
            json_name = f"{opt}_{beta2}_{bs}_{seed}.json"
            
            # Construct the python command exactly matching the sinexperiment sbatch wrap commands
            cmd = (
                f"python sinexperiment/trainer.py "
                f"--optimizer_name {opt} "
                f"--beta1 0.9 "
                f"--beta2 {beta2} "
                f"--batch_size {bs} "
                f"--epochs 300 "
                f"--lr {lr} "
                f"--theta 0.1 "
                f"--sigma 0.3 "
                f"--name_of_json_output_file {json_name} "
                f"--torch_seed {seed} "
                f"--np_seed {seed} "
                f"--random_seed {seed} "
                f"--fisher_save_period 10"
            )
            
            out_file.write(f"echo 'Running Seed {seed}...'\n")
            out_file.write(cmd + "\n\n")

print(f"Success! Generated local execution script: {output_sh_file}")
print(f"To run it, type:\nchmod +x {output_sh_file}\n./{output_sh_file}")
