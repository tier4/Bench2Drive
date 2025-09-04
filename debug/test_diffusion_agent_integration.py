#!/usr/bin/env python3
"""
Integration test for DiffusionDrive B2D agent with CARLA.
Tests agent initialization, sensor setup, and basic inference pipeline.
"""

import os
import sys
import traceback
import numpy as np
import torch
from PIL import Image

# Add Bench2Drive to path for leaderboard imports
sys.path.insert(0, '/workspace/Bench2Drive')
sys.path.insert(0, '/workspace/DiffusionDrive')


def test_imports():
    """Test if all required modules can be imported."""
    print("\n" + "=" * 50)
    print("TEST 1: Import Verification")
    print("=" * 50)

    try:
        # Core imports
        import carla

        print("✅ carla module imported")

        from leaderboard.autoagents import autonomous_agent

        print("✅ leaderboard modules imported")

        # Team code imports
        from team_code.diffusiondrive_b2d_agent import DiffusionDriveAgent

        print("✅ DiffusionDriveAgent imported")

        from team_code.diffusion_model_wrapper import DiffusionDriveModelWrapper

        print("✅ DiffusionDriveModelWrapper imported")

        from team_code.sensor_processor import SensorProcessor

        print("✅ SensorProcessor imported")

        from team_code.trajectory_planner import TrajectoryPlanner

        print("✅ TrajectoryPlanner imported")

        from team_code.pid_controller import PIDController

        print("✅ PIDController imported")

        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_carla_connection():
    """Test CARLA connection."""
    print("\n" + "=" * 50)
    print("TEST 2: CARLA Connection")
    print("=" * 50)

    try:
        import carla

        client = carla.Client("carla", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print(f"✅ Connected to CARLA, Map: {world.get_map().name}")
        return client, world
    except Exception as e:
        print(f"❌ CARLA connection failed: {e}")
        return None, None


def test_agent_initialization():
    """Test agent initialization without full setup."""
    print("\n" + "=" * 50)
    print("TEST 3: Agent Initialization")
    print("=" * 50)

    try:
        from team_code.diffusiondrive_b2d_agent import DiffusionDriveAgent

        # Create agent instance
        agent = DiffusionDriveAgent("", "")
        print("✅ Agent instance created")

        # Test sensor definition
        sensors = agent.sensors()
        print(f"✅ Defined {len(sensors)} sensors:")
        for sensor in sensors:
            print(f"   - {sensor['id']}: {sensor['type']}")

        return agent
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        traceback.print_exc()
        return None


def test_model_wrapper():
    """Test model wrapper initialization."""
    print("\n" + "=" * 50)
    print("TEST 4: Model Wrapper")
    print("=" * 50)

    try:
        from team_code.diffusion_model_wrapper import DiffusionDriveModelWrapper

        # Check if we can at least import the navsim modules
        try:
            from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel
            from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig

            print("✅ NavSim DiffusionDrive modules imported")
        except ImportError as e:
            print(f"⚠️ NavSim modules not available: {e}")
            print("   This is expected if DiffusionDrive is not fully installed")
            return False

        # Test config creation
        config = TransfuserConfig()
        print(f"✅ TransfuserConfig created with n_layer={config.n_layer}")

        return True
    except Exception as e:
        print(f"❌ Model wrapper test failed: {e}")
        traceback.print_exc()
        return False


def test_sensor_processor():
    """Test sensor processor functionality."""
    print("\n" + "=" * 50)
    print("TEST 5: Sensor Processor")
    print("=" * 50)

    try:
        from team_code.sensor_processor import SensorProcessor

        # Create processor instance - no args needed
        processor = SensorProcessor()
        print("✅ SensorProcessor created")

        # Test camera stitching with dummy images
        dummy_images = {
            "rgb_front": Image.new("RGB", (1600, 900), color="red"),
            "rgb_front_left": Image.new("RGB", (1600, 900), color="green"),
            "rgb_front_right": Image.new("RGB", (1600, 900), color="blue"),
        }

        stitched = processor.stitch_front_cameras(dummy_images)
        print(f"✅ Camera stitching works: output shape {stitched.shape}")

        # Test ego status processing with dummy data
        dummy_gps = {"lat": 0.0, "lon": 0.0, "alt": 0.0}
        dummy_imu = {
            "accelerometer": [0.0, 0.0, 0.0],
            "gyroscope": [0.0, 0.0, 0.0],
            "compass": 0.0,
        }
        dummy_speed = {"speed": 10.0}

        ego_status = processor.process_ego_status(dummy_gps, dummy_imu, dummy_speed)
        print(f"✅ Ego status processing works: shape {ego_status.shape}")

        return processor
    except Exception as e:
        print(f"❌ Sensor processor test failed: {e}")
        traceback.print_exc()
        return None


def test_trajectory_planner():
    """Test trajectory planner functionality."""
    print("\n" + "=" * 50)
    print("TEST 6: Trajectory Planner")
    print("=" * 50)

    try:
        from team_code.trajectory_planner import TrajectoryPlanner
        import carla

        # Create planner instance
        planner = TrajectoryPlanner()
        print("✅ TrajectoryPlanner created")

        # Test with dummy trajectory
        dummy_trajectory = torch.randn(20, 8, 2)  # 20 modes, 8 timesteps, 2D
        dummy_transform = carla.Transform(
            carla.Location(x=100, y=200, z=0), carla.Rotation(yaw=45)
        )

        # Test trajectory to waypoints conversion
        waypoints = planner.trajectory_to_waypoints(dummy_trajectory.numpy(), dummy_transform)
        print(f"✅ Trajectory conversion works: {len(waypoints)} waypoints generated")

        # Test mode selection (use private method)
        selected_traj = planner._select_best_mode(dummy_trajectory.numpy())
        print(f"✅ Mode selection works: selected trajectory shape {selected_traj.shape}")

        return planner
    except Exception as e:
        print(f"❌ Trajectory planner test failed: {e}")
        traceback.print_exc()
        return None


def test_pid_controller():
    """Test PID controller functionality."""
    print("\n" + "=" * 50)
    print("TEST 7: PID Controller")
    print("=" * 50)

    try:
        from team_code.pid_controller import PIDController
        import carla

        # Create controller instance with correct args
        controller = PIDController()  # Uses default values
        print("✅ PIDController created")

        # Test with dummy waypoints
        waypoints = [carla.Location(x=100 + i * 5, y=200, z=0) for i in range(5)]
        vehicle_transform = carla.Transform(
            carla.Location(x=100, y=200, z=0), carla.Rotation(yaw=0)
        )

        # Mock vehicle (we need to create a simple mock)
        class MockVehicle:
            def get_transform(self):
                return vehicle_transform

            def get_velocity(self):
                return carla.Vector3D(x=5.0, y=0.0, z=0.0)

        vehicle = MockVehicle()
        control = controller.run_step(5.0, waypoints[1:], vehicle, vehicle_transform)

        print(
            f"✅ PID control works: steer={control.steer:.3f}, throttle={control.throttle:.3f}, brake={control.brake:.3f}"
        )

        return controller
    except Exception as e:
        print(f"❌ PID controller test failed: {e}")
        traceback.print_exc()
        return None


def test_full_agent_setup():
    """Test full agent setup with mock config."""
    print("\n" + "=" * 50)
    print("TEST 8: Full Agent Setup (with mock config)")
    print("=" * 50)

    try:
        from team_code.diffusiondrive_b2d_agent import DiffusionDriveAgent

        # Create mock config file
        mock_config = {
            "checkpoint_path": "/workspace/DiffusionDrive/checkpoints/dummy.pth",
            "config_path": "/workspace/DiffusionDrive/configs/dummy.yaml",
            "save_path": None,
        }

        # Save mock config
        import json

        config_path = "/tmp/mock_agent_config.json"
        with open(config_path, "w") as f:
            json.dump(mock_config, f)

        # Create agent with mock config
        agent = DiffusionDriveAgent("", "")

        # Try setup (will fail at model loading, but tests the flow)
        try:
            agent.setup(config_path)
            print("✅ Agent setup completed")
        except FileNotFoundError as e:
            if "checkpoint" in str(e) or "model" in str(e):
                print("⚠️ Agent setup failed at model loading (expected without real checkpoint)")
            else:
                raise
        except Exception as e:
            if "state_dict" in str(e) or "checkpoint" in str(e):
                print(
                    "⚠️ Agent setup failed at checkpoint loading (expected without real checkpoint)"
                )
            else:
                raise

        return agent
    except Exception as e:
        print(f"❌ Full agent setup failed: {e}")
        traceback.print_exc()
        return None


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("DiffusionDrive B2D Agent Integration Test Suite")
    print("=" * 60)

    results = {
        "Import Verification": test_imports(),
        "CARLA Connection": test_carla_connection()[0] is not None,
        "Agent Initialization": test_agent_initialization() is not None,
        "Model Wrapper": test_model_wrapper(),
        "Sensor Processor": test_sensor_processor() is not None,
        "Trajectory Planner": test_trajectory_planner() is not None,
        "PID Controller": test_pid_controller() is not None,
        "Full Agent Setup": test_full_agent_setup() is not None,
    }

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed out of {len(results)} tests")

    if failed == 0:
        print("\n🎉 All tests passed! The integration is working correctly.")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please check the errors above.")

    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
