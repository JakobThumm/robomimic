#!/bin/bash

# Run all experiments script
# This script runs trained agents on all experiment configurations systematically
# Environments are run sequentially to avoid GPU overload
# Configs within each environment run in parallel

set -e  # Exit on any error

# Base directory
BASE_DIR="/home/jakob/Promotion/code/robomimic"
cd "$BASE_DIR"

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hrgym

# Environment configurations
declare -A ENV_MAP=(
    ["lift"]="LiftHumanEnv"
    ["can"]="PickPlaceCanHumanEnv" 
    ["square"]="NutAssemblySquareHumanEnv"
    ["tool_hang"]="ToolHangHumanEnv"
)

# Available environments (based on models found)
ENVIRONMENTS=("lift" "can" "square" "tool_hang")

# Experiment configuration directories
CONFIG_DIRS=("failsafe_single" "failsafe_waypoints" "osc")

# Function to run a single experiment
run_experiment() {
    local env=$1
    local config_dir=$2
    local config_file=$3
    local human_env=$4
    
    # Extract config name without extension for naming
    local config_name=$(basename "$config_file" .json)
    
    # Define output paths
    local results_dir="$BASE_DIR/results/$env/ph/$config_dir"
    local video_path="$results_dir/video/${config_name}.mp4"
    local csv_path="$results_dir/${config_name}.csv"
    
    # Agent model path
    local agent_path="$BASE_DIR/models/test/$env/ph/last.pth"
    
    echo "Running experiment: $env/$config_dir/$config_name"
    echo "  Agent: $agent_path"
    echo "  Config: $config_file"
    echo "  Environment: $human_env"
    echo "  Results: $csv_path"
    
    # Run the experiment
    python robomimic/scripts/run_trained_agent.py \
        --agent "$agent_path" \
        --n_rollouts 100 \
        --horizon 400 \
        --seed 0 \
        --video_path "$video_path" \
        --video_skip 1 \
        --csv_path "$csv_path" \
        --env "$human_env" \
        --config "$config_file"
        
    echo "Completed: $env/$config_dir/$config_name"
    echo "---"
}

# Function to run all configs in a directory in parallel
run_configs_parallel() {
    local env=$1
    local config_dir=$2
    local human_env=$3
    
    local full_config_dir="$BASE_DIR/experiment_configs/$config_dir"
    
    if [ ! -d "$full_config_dir" ]; then
        echo "Warning: Config directory does not exist: $full_config_dir"
        return
    fi
    
    echo "Starting parallel execution for $env/$config_dir"
    
    # Create array to store background job PIDs
    local pids=()
    
    # Launch all configs in parallel
    for config_file in "$full_config_dir"/*.json; do
        if [ -f "$config_file" ]; then
            run_experiment "$env" "$config_dir" "$config_file" "$human_env" &
            pids+=($!)
        fi
    done
    
    # Wait for all parallel jobs to complete
    echo "Waiting for all $env/$config_dir experiments to complete..."
    for pid in "${pids[@]}"; do
        wait $pid
        if [ $? -ne 0 ]; then
            echo "Error: One of the experiments failed (PID: $pid)"
        fi
    done
    
    echo "All $env/$config_dir experiments completed!"
    echo "========================================"
}

# Main execution
echo "Starting large-scale experiment run"
echo "========================================"

# Check if models exist
for env in "${ENVIRONMENTS[@]}"; do
    model_path="$BASE_DIR/models/test/$env/ph/last.pth"
    if [ ! -f "$model_path" ]; then
        echo "Error: Model not found for $env: $model_path"
        exit 1
    fi
done

echo "All required models found. Starting experiments..."
echo ""

# Run experiments for each environment sequentially
for env in "${ENVIRONMENTS[@]}"; do
    human_env="${ENV_MAP[$env]}"
    echo "========================================"
    echo "Starting experiments for environment: $env"
    echo "Using HumanEnv: $human_env"
    echo "========================================"
    
    # Run each config directory
    for config_dir in "${CONFIG_DIRS[@]}"; do
        run_configs_parallel "$env" "$config_dir" "$human_env"
        echo ""
    done
    
    echo "Completed all experiments for environment: $env"
    echo ""
done

echo "========================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "========================================"

# Generate summary report
echo "Generating summary report..."
echo "Summary Report - $(date)" > "$BASE_DIR/experiment_summary.txt"
echo "=============================" >> "$BASE_DIR/experiment_summary.txt"

for env in "${ENVIRONMENTS[@]}"; do
    echo "" >> "$BASE_DIR/experiment_summary.txt"
    echo "Environment: $env" >> "$BASE_DIR/experiment_summary.txt"
    echo "-------------------" >> "$BASE_DIR/experiment_summary.txt"
    
    for config_dir in "${CONFIG_DIRS[@]}"; do
        echo "  Config Directory: $config_dir" >> "$BASE_DIR/experiment_summary.txt"
        results_dir="$BASE_DIR/results/$env/ph/$config_dir"
        if [ -d "$results_dir" ]; then
            csv_count=$(find "$results_dir" -name "*.csv" | wc -l)
            echo "    CSV files generated: $csv_count" >> "$BASE_DIR/experiment_summary.txt"
        fi
    done
done

echo "Summary report saved to: $BASE_DIR/experiment_summary.txt"
echo "All results are stored in: $BASE_DIR/results/"