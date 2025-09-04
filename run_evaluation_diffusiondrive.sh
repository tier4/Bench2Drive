#!/bin/bash
# DiffusionDrive evaluation script for Bench2Drive
# Single GPU execution

# Environment setup
export CARLA_ROOT=/workspace/Bench2Drive/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export PYTHONPATH=$PYTHONPATH:/workspace/DiffusionDrive

export SCENARIO_RUNNER_ROOT=scenario_runner
export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export IS_BENCH2DRIVE=True

# Configuration
PORT=2000
TM_PORT=8000
DEBUG_CHALLENGE=0
REPETITIONS=1
RESUME=False # Set to False for fresh evaluation, True to continue from checkpoint

# Routes configuration - can be changed for different tests
ROUTES=${1:-test_routes_3maps.xml} # Default to test routes, can override with argument

# Timeout configuration (in seconds)
# Default: 600s (10 minutes) client timeout
# Can be overridden with second argument
TIMEOUT=${2:-600}
echo "Using routes: $ROUTES"
echo "Using timeout: ${TIMEOUT}s"

# Agent configuration
TEAM_AGENT=leaderboard/team_code/diffusiondrive_b2d_agent.py
TEAM_CONFIG=test_files/exp0_datasetv2_archv0_bs256x8_ep3000_lr1e-5_step2539.ckpt
# TEAM_CONFIG=test_files/test_b2d.ckpt

# Output configuration
ALGO=diffusiondrive
PLANNER_TYPE=traj
# SAVE_PATH=./eval_results_${ALGO}_$(date +%Y%m%d_%H%M%S)
SAVE_PATH=/mnt/nvme1/result/eval_results_${ROUTES}_${ALGO}_$(date +%Y%m%d_%H%M%S)

# Create output directory first
mkdir -p $SAVE_PATH

# Now set checkpoint path inside the results directory
CHECKPOINT_ENDPOINT=${SAVE_PATH}/eval_${ALGO}_${ROUTES%.xml}.json

# Enable image saving for video generation (DiffusionDrive agent checks this)
export SAVE_PATH=$SAVE_PATH
export ROUTES=$ROUTES # Agent needs this for save path creation

# GPU configuration
GPU_RANK=${GPU_RANK:-0} # Use GPU 0 by default, can override with environment variable

echo "========================================="
echo "DiffusionDrive Evaluation Configuration"
echo "========================================="
echo "Routes: $ROUTES"
echo "Agent: $TEAM_AGENT"
echo "Config: $TEAM_CONFIG"
echo "Output: $SAVE_PATH"
echo "Checkpoint: $CHECKPOINT_ENDPOINT"
echo "GPU: $GPU_RANK"
echo "CARLA: carla:$PORT"
echo "========================================="

# Run evaluation
CUDA_VISIBLE_DEVICES=${GPU_RANK} python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
    --routes=${ROUTES} \
    --repetitions=${REPETITIONS} \
    --track=${CHALLENGE_TRACK_CODENAME} \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --debug=${DEBUG_CHALLENGE} \
    --resume=${RESUME} \
    --host=carla \
    --port=${PORT} \
    --traffic-manager-port=${TM_PORT} \
    --gpu-rank=${GPU_RANK} 2>&1 | tee ${SAVE_PATH}/evaluation.log

echo "========================================="
echo "Evaluation completed!"
echo "Results saved to: $SAVE_PATH"
echo "Checkpoint: $CHECKPOINT_ENDPOINT"
echo "========================================="
