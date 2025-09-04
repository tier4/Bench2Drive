#!/usr/bin/env python3
"""
Direct evaluation runner with better error handling.
"""
import os
import sys
import traceback

# Set environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['IS_BENCH2DRIVE'] = 'True'
os.environ['CARLA_ROOT'] = '/workspace/Bench2Drive/carla'
os.environ['SCENARIO_RUNNER_ROOT'] = '/workspace/Bench2Drive/scenario_runner'
os.environ['LEADERBOARD_ROOT'] = '/workspace/Bench2Drive/leaderboard'
os.environ['CHALLENGE_TRACK_CODENAME'] = 'SENSORS'

# Set paths
sys.path.insert(0, '/workspace/Bench2Drive')
sys.path.insert(0, '/workspace/Bench2Drive/leaderboard')
sys.path.insert(0, '/workspace/Bench2Drive/leaderboard/team_code')
sys.path.insert(0, '/workspace/Bench2Drive/scenario_runner')
sys.path.insert(0, '/workspace/DiffusionDrive')
sys.path.insert(0, '/workspace/Bench2Drive/carla/PythonAPI')
sys.path.insert(0, '/workspace/Bench2Drive/carla/PythonAPI/carla')

print("Starting DiffusionDrive evaluation...")
print("-" * 50)

# Build command line arguments
sys.argv = [
    'run_eval_direct.py',
    '--host=carla',
    '--port=2000',
    '--traffic-manager-port=8000',
    '--routes=test_route_town10hd.xml',  # Single route for Town10HD
    '--repetitions=1',
    '--track=SENSORS',
    '--checkpoint=eval_diffusiondrive.json',
    '--agent=leaderboard/team_code/diffusiondrive_b2d_agent.py',
    '--agent-config=test_files/diffusiondrive_config.yaml+test_files/test_b2d.ckpt+test_run',
    '--debug=0',
    '--resume=True',
    '--timeout=9000',
]

try:
    print("Importing evaluator...")
    from leaderboard.leaderboard_evaluator import main
    
    print("Starting evaluation...")
    main()
    print("Evaluation completed successfully!")
    
except Exception as e:
    print(f"\nError occurred: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)