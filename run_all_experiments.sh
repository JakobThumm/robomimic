#!/bin/bash

# Run all experiments script
# This script runs trained agents on all experiment configurations systematically
# Can run all environments or a specific environment using --name parameter
# Usage: ./run_all_experiments.sh [--name ENVIRONMENT_NAME] [--parallel]

set +e  # Don't exit on errors - we want to continue with other experiments

# Base directory - get the directory where this script is located
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

# Environment configurations
declare -A ENV_MAP=(
    ["lift"]="LiftHumanEnv"
    ["can"]="PickPlaceCanHumanEnv" 
    ["square"]="NutAssemblySquareHumanEnv"
    ["tool_hang"]="ToolHangHumanEnv"
)

# Available environments (based on models found)
ENVIRONMENTS=("lift" "can" "square" "tool_hang")

# Parse command line arguments
SPECIFIC_ENV=""
PARALLEL_MODE=false
CONDA_PATH="~/anaconda3"
while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            SPECIFIC_ENV="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL_MODE=true
            shift
            ;;
        --conda-path)
            CONDA_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 [--name ENVIRONMENT_NAME] [--parallel] [--conda-path PATH]"
            echo "Available environments: ${ENVIRONMENTS[*]}"
            echo "Options:"
            echo "  --name ENV_NAME        Run experiments only for specific environment"
            echo "  --parallel             Run all experiments in parallel (default: sequential)"
            echo "  --conda-path PATH      Path to conda installation (default: ~/anaconda3)"
            exit 1
            ;;
    esac
done

# If specific environment is provided, validate and use it
if [ -n "$SPECIFIC_ENV" ]; then
    if [[ ! " ${ENVIRONMENTS[*]} " =~ " ${SPECIFIC_ENV} " ]]; then
        echo "Error: Environment '$SPECIFIC_ENV' is not valid."
        echo "Available environments: ${ENVIRONMENTS[*]}"
        exit 1
    fi
    ENVIRONMENTS=("$SPECIFIC_ENV")
    echo "Running experiments for environment: $SPECIFIC_ENV"
else
    echo "Running experiments for all environments: ${ENVIRONMENTS[*]}"
fi

# Display execution mode
if [ "$PARALLEL_MODE" = true ]; then
    echo "Execution mode: PARALLEL (all experiments run simultaneously)"
    echo "Warning: Parallel mode will use significant system resources"
else
    echo "Execution mode: SEQUENTIAL (one experiment at a time)"
fi

# Activate conda environment
echo "Debug: CONDA_PATH = '${CONDA_PATH}'"

# Expand tilde in conda path if present, otherwise use as-is
if [[ "${CONDA_PATH}" == ~* ]]; then
    CONDA_PATH_EXPANDED="${CONDA_PATH/#\~/$HOME}"
else
    CONDA_PATH_EXPANDED="${CONDA_PATH}"
fi

echo "Using conda path: ${CONDA_PATH_EXPANDED}"

if [ -f "${CONDA_PATH_EXPANDED}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_PATH_EXPANDED}/etc/profile.d/conda.sh"
    conda activate hrgym
else
    echo "Error: conda.sh not found at ${CONDA_PATH_EXPANDED}/etc/profile.d/conda.sh"
    echo "Please check your conda installation path."
    exit 1
fi

# Experiment configuration directories
CONFIG_DIRS=("failsafe_single" "failsafe_waypoints" "osc" "cbf")

# Function to run a single experiment
run_experiment() {
    local env=$1
    local config_dir=$2
    local config_file=$3
    local human_env=$4
    local error_log_path=$5
    
    # Extract config name without extension for naming
    local config_name=$(basename "$config_file" .json)
    
    # Define output paths
    local results_dir="$BASE_DIR/results_selection/$env/ph/$config_dir"
    local video_path="$results_dir/video/${config_name}.mp4"
    local csv_path="$results_dir/${config_name}.csv"
    
    # Agent model path
    local agent_path="$BASE_DIR/models/test/$env/ph/last.pth"
    
    echo "Running experiment: $env/$config_dir/$config_name"
    echo "  Agent: $agent_path"
    echo "  Config: $config_file"
    echo "  Environment: $human_env"
    echo "  Results: $csv_path"
    
    # Run the experiment and capture exit code
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
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "SUCCESS: $env/$config_dir/$config_name"
    else
        echo "FAILED: $env/$config_dir/$config_name (exit code: $exit_code)"
        # Log failure to error log
        echo "$(date): FAILED - $env/$config_dir/$config_name (exit code: $exit_code)" >> "$error_log_path"
    fi
    echo "---"
    
    return $exit_code
}

# Function to run all configs in a directory in parallel
run_configs_parallel() {
    local env=$1
    local config_dir=$2
    local human_env=$3
    local error_log_path=$4
    
    local full_config_dir="$BASE_DIR/experiment_configs_selection/$config_dir"
    
    if [ ! -d "$full_config_dir" ]; then
        echo "Warning: Config directory does not exist: $full_config_dir"
        return 1
    fi
    
    echo "Starting parallel execution for $env/$config_dir"
    echo "Full config dir $full_config_dir"
    
    # Count total configs
    local total_configs=$(find "$full_config_dir" -name "*.json" -type f | wc -l)
    
    if [ $total_configs -eq 0 ]; then
        echo "No JSON config files found in $full_config_dir"
        return 1
    fi
    
    echo "Found $total_configs configurations to run in parallel"
    echo "---"
    
    # Array to store background process PIDs
    local pids=()
    
    # Start all experiments in parallel
    for config_file in "$full_config_dir"/*.json; do
        if [ -f "$config_file" ]; then
            config_name=$(basename "$config_file" .json)
            echo "Starting parallel experiment: $config_name"
            
            # Run experiment in background and capture PID
            (
                echo "========================================"
                echo "Parallel Experiment: $config_name"
                echo "Environment: $env | Config Directory: $config_dir"
                echo "Started at: $(date)"
                echo "========================================"
                
                start_time=$(date +%s)
                
                if run_experiment "$env" "$config_dir" "$config_file" "$human_env" "$error_log_path"; then
                    end_time=$(date +%s)
                    duration=$((end_time - start_time))
                    echo "SUCCESS: $config_name completed in ${duration} seconds"
                else
                    end_time=$(date +%s)
                    duration=$((end_time - start_time))
                    echo "FAILED: $config_name failed after ${duration} seconds"
                fi
            ) &
            
            # Store the PID
            pids+=($!)
        fi
    done
    
    echo "All $total_configs experiments started in parallel"
    echo "Waiting for completion..."
    
    # Wait for all background processes to complete
    local successful_experiments=0
    local failed_experiments=0
    
    for pid in "${pids[@]}"; do
        if wait "$pid"; then
            successful_experiments=$((successful_experiments+1))
        else
            failed_experiments=$((failed_experiments+1))
        fi
    done
    
    echo "All $env/$config_dir experiments completed!"
    echo "Final results: $successful_experiments successful, $failed_experiments failed"
    echo "========================================"
    
    return 0
}

# Function to run all configs in a directory sequentially
run_configs_sequential() {
    local env=$1
    local config_dir=$2
    local human_env=$3
    local error_log_path=$4
    
    local full_config_dir="$BASE_DIR/experiment_configs_selection/$config_dir"
    
    if [ ! -d "$full_config_dir" ]; then
        echo "Warning: Config directory does not exist: $full_config_dir"
        return 1
    fi
    
    echo "Starting sequential execution for $env/$config_dir"
    echo "Full config dir $full_config_dir"
    
    # Count total configs for progress tracking
    local total_configs=$(find "$full_config_dir" -name "*.json" -type f | wc -l)
    local current_config=0
    local failed_experiments=0
    local successful_experiments=0
    
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
            
            # Run experiment and track success/failure
            if run_experiment "$env" "$config_dir" "$config_file" "$human_env" "$error_log_path"; then
                successful_experiments=$((successful_experiments+1))
            else
                failed_experiments=$((failed_experiments+1))
                echo "Experiment failed but continuing with remaining experiments..."
            fi
            
            # Calculate duration
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            
            echo "Experiment completed in ${duration} seconds"
            echo "Progress: $current_config/$total_configs experiments completed in $env/$config_dir"
            echo "Success: $successful_experiments | Failed: $failed_experiments"
            echo ""
        fi
    done
    
    echo "All $env/$config_dir experiments completed!"
    echo "Final results: $successful_experiments successful, $failed_experiments failed"
    echo "========================================"
    
    return 0  # Always return success so the overall script continues
}

# Main execution
echo "Starting large-scale experiment run"
echo "========================================"

# Initialize error log
if [ -n "$SPECIFIC_ENV" ]; then
    error_log="$BASE_DIR/experiment_errors_${SPECIFIC_ENV}.log"
else
    error_log="$BASE_DIR/experiment_errors.log"
fi
echo "Experiment Error Log - Started $(date)" > "$error_log"
echo "=========================================" >> "$error_log"

# Check if models exist
for env in "${ENVIRONMENTS[@]}"; do
    model_path="$BASE_DIR/models/test/$env/ph/last.pth"
    if [ ! -f "$model_path" ]; then
        echo "Error: Model not found for $env: $model_path"
        echo "$(date): CRITICAL - Model not found for $env: $model_path" >> "$error_log"
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
        if [ "$PARALLEL_MODE" = true ]; then
            run_configs_parallel "$env" "$config_dir" "$human_env" "$error_log"
        else
            run_configs_sequential "$env" "$config_dir" "$human_env" "$error_log"
        fi
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
if [ -n "$SPECIFIC_ENV" ]; then
    summary_file="$BASE_DIR/experiment_summary_${SPECIFIC_ENV}.txt"
else
    summary_file="$BASE_DIR/experiment_summary.txt"
fi

echo "Summary Report - $(date)" > "$summary_file"
echo "=============================" >> "$summary_file"

for env in "${ENVIRONMENTS[@]}"; do
    echo "" >> "$summary_file"
    echo "Environment: $env" >> "$summary_file"
    echo "-------------------" >> "$summary_file"
    
    for config_dir in "${CONFIG_DIRS[@]}"; do
        echo "  Config Directory: $config_dir" >> "$summary_file"
        results_dir="$BASE_DIR/results_selection/$env/ph/$config_dir"
        if [ -d "$results_dir" ]; then
            csv_count=$(find "$results_dir" -name "*.csv" | wc -l)
            echo "    CSV files generated: $csv_count" >> "$summary_file"
        fi
    done
done

echo "Summary report saved to: $summary_file"
echo "All results are stored in: $BASE_DIR/results_selection/"

# Display error summary if there were any failures
if [ -f "$error_log" ] && [ -s "$error_log" ]; then
    error_count=$(grep -c "FAILED -" "$error_log" 2>/dev/null || echo "0")
    if [ "$error_count" -gt 0 ]; then
        echo ""
        echo "Warning: $error_count experiments failed during execution"
        echo "Check error log for details: $error_log"
    else
        echo ""
        echo "All experiments completed successfully!"
    fi
else
    echo ""
    echo "All experiments completed successfully!"
fi
