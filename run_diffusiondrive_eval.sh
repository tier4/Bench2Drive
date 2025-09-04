#!/bin/bash
# Proper evaluation script for DiffusionDrive with all paths set correctly

# Set CARLA paths - using the local carla directory in the project
export CARLA_ROOT=/workspace/Bench2Drive/carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg

# Add Bench2Drive paths
export PYTHONPATH=$PYTHONPATH:/workspace/Bench2Drive
export PYTHONPATH=$PYTHONPATH:/workspace/Bench2Drive/leaderboard
export PYTHONPATH=$PYTHONPATH:/workspace/Bench2Drive/leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:/workspace/Bench2Drive/scenario_runner

# Add DiffusionDrive path
export PYTHONPATH=$PYTHONPATH:/workspace/DiffusionDrive

# Set environment variables
export SCENARIO_RUNNER_ROOT=/workspace/Bench2Drive/scenario_runner
export LEADERBOARD_ROOT=/workspace/Bench2Drive/leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export IS_BENCH2DRIVE=True
export DEBUG_CHALLENGE=0
export REPETITIONS=1
export RESUME=True

# Evaluation parameters - CARLA is running on port 2000
PORT=${PORT:-2000}
TM_PORT=${TM_PORT:-8000}
GPU_RANK=${GPU_RANK:-0}

# Enable CUDA for GPU acceleration
export CUDA_VISIBLE_DEVICES="0"  # Use GPU 0
ROUTES="leaderboard/data/drivetransformer_bench2drive_dev10.xml"  # Use Dev10 for testing (10 routes)
TEAM_AGENT="leaderboard/team_code/diffusiondrive_b2d_agent.py"
TEAM_CONFIG="test_files/diffusiondrive_config.yaml+test_files/test_b2d.ckpt+test_run"
CHECKPOINT_ENDPOINT="eval_diffusiondrive.json"
SAVE_PATH="./eval_diffusiondrive/"
PLANNER_TYPE="only_traj"

echo "============================================"
echo "Running DiffusionDrive Evaluation"
echo "============================================"
echo "CARLA_ROOT: $CARLA_ROOT"
echo "Agent: $TEAM_AGENT"
echo "Config: $TEAM_CONFIG"
echo "GPU: $GPU_RANK"
echo "Port: $PORT"
echo "Routes: $ROUTES"
echo "Save Path: $SAVE_PATH"
echo "============================================"

# Create save directory
mkdir -p $SAVE_PATH

# Change to Bench2Drive directory
cd /workspace/Bench2Drive

# Run the evaluation directly with Python
# Note: Some parameters are optional and will be empty if not set
# CARLA is running at hostname 'carla' (Docker setup)
python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
    --host=carla \
    --routes=${ROUTES} \
    --repetitions=${REPETITIONS} \
    --track=${CHALLENGE_TRACK_CODENAME} \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --debug=${DEBUG_CHALLENGE} \
    --resume=${RESUME} \
    --port=${PORT} \
    --timeout=9000 \
    --trafficManagerPort=${TM_PORT}