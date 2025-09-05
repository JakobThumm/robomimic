#!/bin/bash

# Run all experiments script
# This script runs trained agents on all experiment configurations systematically
# All experiments run sequentially for better visibility and control

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
ENVIRONMENTS=("lift" "can" "square") #"tool_hang")

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

# Function to run all configs in a directory sequentially
run_configs_sequential() {
    local env=$1
    local config_dir=$2
    local human_env=$3
    
    local full_config_dir="$BASE_DIR/experiment_configs/$config_dir"
    
    if [ ! -d "$full_config_dir" ]; then
        echo "Warning: Config directory does not exist: $full_config_dir"
        return
    fi
    
    echo "Starting sequential execution for $env/$config_dir"
    echo ""
    
    # Count total configs for progress tracking
    local total_configs=$(find "$full_config_dir" -name "*.json" -type f | wc -l)
    local current_config=0
    echo "---"
    # Run all configs sequentially
    for config_file in "$full_config_dir"/*.json; do
        echo "Looking for $config_file"
        if [ -f "$config_file" ]; then
            echo "$config_file found"
            current_config=$((current_config+1))
            echo "Running config number $current_config"
            config_name=$(basename "$config_file" .json)
            
            echo "========================================"
            echo "Experiment $current_config/$total_configs: $config_name"
            echo "Environment: $env | Config Directory: $config_dir"
            echo "Started at: $(date)"
            echo "========================================"
            
            # Record start time
            start_time=$(date +%s)
            
            run_experiment "$env" "$config_dir" "$config_file" "$human_env"
            
            # Calculate duration
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            
            echo "Experiment completed in ${duration} seconds"
            echo "Progress: $current_config/$total_configs experiments completed in $env/$config_dir"
            echo ""
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
        run_configs_sequential "$env" "$config_dir" "$human_env"
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