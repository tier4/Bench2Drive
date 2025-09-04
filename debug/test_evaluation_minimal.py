#!/usr/bin/env python3
"""
Minimal evaluation test to identify segfault location.
"""

import os
import sys
import argparse

# Set paths
sys.path.insert(0, '/workspace/Bench2Drive')
sys.path.insert(0, '/workspace/Bench2Drive/leaderboard')
sys.path.insert(0, '/workspace/Bench2Drive/scenario_runner')
sys.path.insert(0, '/workspace/DiffusionDrive')
sys.path.insert(0, '/workspace/Bench2Drive/carla/PythonAPI')
sys.path.insert(0, '/workspace/Bench2Drive/carla/PythonAPI/carla')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['IS_BENCH2DRIVE'] = 'True'
os.environ['CARLA_ROOT'] = '/workspace/Bench2Drive/carla'
os.environ['SCENARIO_RUNNER_ROOT'] = '/workspace/Bench2Drive/scenario_runner'
os.environ['LEADERBOARD_ROOT'] = '/workspace/Bench2Drive/leaderboard'
os.environ['CHALLENGE_TRACK_CODENAME'] = 'SENSORS'

print("[1] Importing modules...")

try:
    import torch
    print(f"   PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
except Exception as e:
    print(f"   PyTorch import failed: {e}")

try:
    import carla
    print("   CARLA imported")
except Exception as e:
    print(f"   CARLA import failed: {e}")

try:
    from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
    print("   LeaderboardEvaluator imported")
except Exception as e:
    print(f"   LeaderboardEvaluator import failed: {e}")
    sys.exit(1)

print("\n[2] Creating evaluator arguments...")

# Minimal arguments for evaluator
args = argparse.Namespace(
    host='carla',
    port=2000,
    trafficManagerPort=8000,
    debug=0,
    routes='leaderboard/data/drivetransformer_bench2drive_dev10.xml',
    scenarios='leaderboard/data/scenarios/no_scenarios.json',
    repetitions=1,
    checkpoint='eval_diffusiondrive.json',
    agent='leaderboard/team_code/diffusiondrive_b2d_agent.py',
    agent_config='test_files/diffusiondrive_config.yaml+test_files/test_b2d.ckpt+test_run',
    track='SENSORS',
    resume=True,
    timeout=9000,
    gpu_rank=0
)

print(f"   Host: {args.host}:{args.port}")
print(f"   Agent: {args.agent}")
print(f"   Routes: {args.routes}")

print("\n[3] Creating LeaderboardEvaluator instance...")

try:
    evaluator = LeaderboardEvaluator(args, statistics_manager=None)
    print("   Evaluator created successfully")
except Exception as e:
    print(f"   Failed to create evaluator: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[4] Starting evaluation run...")
print("   If segfault occurs here, it's in the evaluation loop")

try:
    evaluator.run(args)
    print("   Evaluation completed!")
except KeyboardInterrupt:
    print("\n   Evaluation interrupted by user")
except Exception as e:
    print(f"   Evaluation failed: {e}")
    import traceback
    traceback.print_exc()