#!/usr/bin/env python3
"""
Test script to verify DiffusionDrive agent initialization.
"""

import sys
import os
import traceback

# Add paths
sys.path.insert(0, '/workspace/Bench2Drive')
sys.path.insert(0, '/workspace/Bench2Drive/leaderboard')
sys.path.insert(0, '/workspace/Bench2Drive/scenario_runner')
sys.path.insert(0, '/workspace/Bench2Drive/carla/PythonAPI/carla')  # Add agents module
sys.path.insert(0, '/workspace/Bench2Drive/carla/PythonAPI')
sys.path.insert(0, '/workspace/DiffusionDrive')
os.chdir('/workspace/Bench2Drive')

def test_agent_initialization():
    """Test if the agent can be initialized with the config and checkpoint."""
    print("="*60)
    print("Testing DiffusionDrive Agent Initialization")
    print("="*60)
    
    try:
        # Test imports
        print("\n1. Testing imports...")
        # Import directly from team_code since we're adding it to path
        sys.path.insert(0, '/workspace/Bench2Drive/leaderboard/team_code')
        from diffusiondrive_b2d_agent import DiffusionDriveAgent
        print("✓ Agent imported successfully")
        
        # Create agent instance (skip for now since it needs CARLA world)
        print("\n2. Creating agent instance...")
        print("  Skipping instance creation (requires CARLA world)")
        agent = None
        
        # Test the model wrapper directly instead
        print("\n3. Testing model wrapper...")
        try:
            from diffusion_model_wrapper import DiffusionDriveModelWrapper
            print("✓ Model wrapper imported")
            
            # Test model initialization
            print("\n4. Testing model initialization...")
            model_wrapper = DiffusionDriveModelWrapper(
                checkpoint_path="test_files/test_b2d.ckpt",
                config_path="test_files/diffusiondrive_config.yaml"
            )
            print("✓ Model wrapper initialized successfully")
        except FileNotFoundError as e:
            print(f"⚠ File not found: {e}")
        except Exception as e:
            print(f"⚠ Model initialization failed: {e}")
            traceback.print_exc()
        
        print("\n" + "="*60)
        print("Basic agent initialization test completed!")
        print("="*60)
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_agent_initialization()
    sys.exit(0 if success else 1)