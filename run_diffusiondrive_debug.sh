#!/bin/bash
# Debug evaluation script for DiffusionDrive agent

BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/bench2drive220
TEAM_AGENT=leaderboard/team_code/diffusiondrive_b2d_agent.py
TEAM_CONFIG=test_files/diffusiondrive_config.yaml+test_files/test_b2d.ckpt+test_run
BASE_CHECKPOINT_ENDPOINT=eval_diffusiondrive
SAVE_PATH=./eval_diffusiondrive/
PLANNER_TYPE=only_traj

# Use GPU 0 by default, can be changed
GPU_RANK=${GPU_RANK:-0}
PORT=$BASE_PORT
TM_PORT=$BASE_TM_PORT
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}.json"

echo "============================================"
echo "Running DiffusionDrive Debug Evaluation"
echo "============================================"
echo "Agent: $TEAM_AGENT"
echo "Config: $TEAM_CONFIG"
echo "GPU: $GPU_RANK"
echo "Port: $PORT"
echo "Routes: $ROUTES"
echo "Save Path: $SAVE_PATH"
echo "============================================"

# Create save directory if it doesn't exist
mkdir -p $SAVE_PATH

# Run the evaluation
bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK