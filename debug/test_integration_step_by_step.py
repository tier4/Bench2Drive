#!/usr/bin/env python3
"""
Step-by-step integration test to identify where the segfault occurs.
Run this after installing any new dependencies.
"""

import os
import sys
import traceback

# Set up paths
sys.path.insert(0, '/workspace/Bench2Drive')
sys.path.insert(0, '/workspace/Bench2Drive/leaderboard')
sys.path.insert(0, '/workspace/Bench2Drive/leaderboard/team_code')
sys.path.insert(0, '/workspace/Bench2Drive/scenario_runner')
sys.path.insert(0, '/workspace/DiffusionDrive')
sys.path.insert(0, '/workspace/Bench2Drive/carla/PythonAPI')
sys.path.insert(0, '/workspace/Bench2Drive/carla/PythonAPI/carla')

print("="*60)
print("STEP-BY-STEP INTEGRATION TEST")
print("="*60)

# Test 1: PyTorch and CUDA
print("\n[1] Testing PyTorch and CUDA...")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ PyTorch/CUDA failed: {e}")
    traceback.print_exc()

# Test 2: CARLA import
print("\n[2] Testing CARLA import...")
try:
    import carla
    print("✓ CARLA module imported")
except Exception as e:
    print(f"✗ CARLA import failed: {e}")
    traceback.print_exc()

# Test 3: Leaderboard evaluator
print("\n[3] Testing leaderboard evaluator import...")
try:
    import leaderboard.leaderboard_evaluator as evaluator
    print("✓ Leaderboard evaluator imported")
except Exception as e:
    print(f"✗ Evaluator import failed: {e}")
    traceback.print_exc()

# Test 4: Autonomous agent base
print("\n[4] Testing autonomous agent base...")
try:
    from leaderboard.autoagents import autonomous_agent
    print("✓ Autonomous agent base imported")
except Exception as e:
    print(f"✗ Autonomous agent import failed: {e}")
    traceback.print_exc()

# Test 5: DiffusionDrive agent
print("\n[5] Testing DiffusionDrive agent import...")
try:
    from team_code.diffusiondrive_b2d_agent import DiffusionDriveAgent
    print("✓ DiffusionDrive agent imported")
except Exception as e:
    print(f"✗ DiffusionDrive agent import failed: {e}")
    traceback.print_exc()

# Test 6: Model wrapper
print("\n[6] Testing model wrapper...")
try:
    from team_code.diffusion_model_wrapper import DiffusionDriveModelWrapper
    print("✓ Model wrapper imported")
    
    # Try to initialize with test files
    try:
        wrapper = DiffusionDriveModelWrapper(
            checkpoint_path="test_files/test_b2d.ckpt",
            config_path="test_files/diffusiondrive_config.yaml"
        )
        print("✓ Model wrapper initialized")
    except FileNotFoundError as e:
        print(f"⚠ Model files not found: {e}")
    except Exception as e:
        print(f"⚠ Model initialization failed: {e}")
except Exception as e:
    print(f"✗ Model wrapper import failed: {e}")
    traceback.print_exc()

# Test 7: CARLA connection
print("\n[7] Testing CARLA connection...")
try:
    client = carla.Client("carla", 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    print(f"✓ Connected to CARLA")
    print(f"✓ Map: {world.get_map().name}")
except Exception as e:
    print(f"✗ CARLA connection failed: {e}")

# Test 8: Combined PyTorch + CARLA
print("\n[8] Testing PyTorch + CARLA together...")
try:
    import torch
    import carla
    
    # Create a dummy tensor
    tensor = torch.randn(1, 3, 224, 224)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
        print("✓ Created CUDA tensor")
    else:
        print("✓ Created CPU tensor")
    
    # Access CARLA
    client = carla.Client("carla", 2000)
    client.set_timeout(2.0)
    print("✓ CARLA client created")
    
    print("✓ PyTorch + CARLA work together")
except Exception as e:
    print(f"✗ PyTorch + CARLA failed: {e}")
    traceback.print_exc()

# Test 9: Agent setup simulation
print("\n[9] Testing agent setup simulation...")
try:
    from team_code.diffusiondrive_b2d_agent import DiffusionDriveAgent
    
    # Create agent instance
    agent = DiffusionDriveAgent("", "")
    print("✓ Agent instance created")
    
    # Get sensor configuration
    sensors = agent.sensors()
    print(f"✓ Got {len(sensors)} sensor configs")
    
    # Test config parsing
    config_string = "test_files/diffusiondrive_config.yaml+test_files/test_b2d.ckpt+test_run"
    parts = config_string.split('+')
    print(f"✓ Config parsed: {len(parts)} parts")
    
except Exception as e:
    print(f"✗ Agent setup failed: {e}")
    traceback.print_exc()

print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print("\nIf all tests pass, the segfault is likely occurring during:")
print("1. The actual evaluation loop")
print("2. When agent.setup() is called with the full CARLA world")
print("3. During sensor data processing in the main loop")
print("\nNext steps:")
print("- Add more logging to the agent's setup() method")
print("- Test with minimal route (single waypoint)")
print("- Use gdb to get exact crash location")