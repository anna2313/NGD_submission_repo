import json

# ==========================================
# Configuration
# ==========================================
json_file_path = "linearregexperiment/saved_results/best_learning_rates_selected_optimizers.json" # <-- UPDATE THIS
output_sh_file = "run_best_linear_experiments_local.sh"

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
    out_file.write("mkdir -p linearregexperiment/logs\n\n")
    
    out_file.write("echo 'Starting Initialization...'\n")
    out_file.write("python linearregexperiment/initialization.py\n\n")
    
    out_file.write("echo 'Starting Training Experiments...'\n\n")

    # Loop through the best configurations from the JSON
    for config in data["best_configs_above_threshold"]:
        opt = config["optimizer"]
        beta2 = config["beta2"]
        bs = config["batch_size"]
        lr = config["best_learning_rate"]
        
        out_file.write(f"# === {opt} | Beta2: {beta2} | Batch Size: {bs} | Best LR: {lr} ===\n")
        
        # Generate a run for seeds 0 through 4
        for seed in range(5):
            json_name = f"{opt}_{beta2}_{bs}_{seed}.json"
            
            # Construct the python command
            cmd = (
                f"python linearregexperiment/trainer.py "
                f"--optimizer_name {opt} "
                f"--beta1 0.9 "
                f"--beta2 {beta2} "
                f"--batch_size {bs} "
                f"--epochs 300 "
                f"--lr {lr} "
                f"--theta 1.0 2.0 "
                f"--sigma 0.01 "
                f"--name_of_json_output_file {json_name} "
                f"--torch_seed {seed} "
                f"--np_seed {seed} "
                f"--random_seed {seed} "
                f"--initialization_type normal "
                f"--moving_average_fishers_on "
                f"--fisher_save_period 10"
            )
            
            out_file.write(f"echo 'Running Seed {seed}...'\n")
            out_file.write(cmd + "\n\n")

print(f"Success! Generated local execution script: {output_sh_file}")
