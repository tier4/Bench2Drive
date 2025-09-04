# DiffusionDrive Usage Guide for Bench2Drive

**✅ STATUS (2025-08-28)**: All critical issues fixed. Model runs successfully with navigation commands.

## Quick Start

### 1. Closed-Loop Evaluation on Small Subset (4 Routes)

```bash
# Run evaluation on 4 Town10HD routes for quick testing
./run_evaluation_diffusiondrive.sh test_routes_town10hd_only.xml

# The script automatically:
# - Creates timestamped results directory
# - Saves images every 10 frames
# - Generates evaluation metrics
# - Stores trajectory predictions in metadata
```

### 2. Full 220-Route Evaluation

```bash
# Run complete Bench2Drive evaluation (12-16 hours on single GPU)
./run_evaluation_diffusiondrive.sh bench2drive220.xml

# Results will be saved in: eval_results_diffusiondrive_[timestamp]/
```

### 3. Visualize Results

#### Generate Video with Control Overlay

```bash
# Basic video with navigation commands and control info
python3 generate_video_ddrive.py -f eval_results_diffusiondrive_*/[route_folder] -o output.mp4
```

#### Generate BEV Video with Trajectory

```bash
# Bird's eye view with predicted trajectory visualization
python3 generate_bev_video_ddrive.py -f eval_results_diffusiondrive_*/[route_folder] -o output_bev.mp4
```

## Configuration

### Model Checkpoint

- Location: `/workspace/Bench2Drive/test_files/test_b2d.ckpt`
- Anchor file: `/workspace/Bench2Drive/test_files/kmeans_b2d_v2_traj_20.npy`

### Environment Variables (set automatically by script)

- `ROUTES`: XML file with route definitions
- `SAVE_PATH`: Directory for saving evaluation results
- `CHECKPOINT_ENDPOINT`: JSON file for resuming interrupted evaluations
- `RESUME`: Set to `False` for fresh runs, `True` to continue

### Key Parameters in Script

```bash
ALGO="diffusiondrive"
TEAM_AGENT="leaderboard/team_code/diffusiondrive_b2d_agent.py"
TEAM_CONFIG="test_files/diffusiondrive_config.yaml+test_files/test_b2d.ckpt"
PORT=2000  # CARLA server port
TM_PORT=2500  # Traffic manager port
```

## Output Structure

```
eval_results_diffusiondrive_[timestamp]/
├── eval_diffusiondrive_[routes].json  # Evaluation checkpoint
├── evaluation.log                      # Detailed logs
└── [route_name]_[timestamp]/
    ├── rgb_front/        # Front camera images
    ├── rgb_front_left/   # Multi-view camera images
    ├── rgb_front_right/
    ├── rgb_back/
    ├── rgb_back_left/
    ├── rgb_back_right/
    ├── bev/             # Bird's eye view (currently empty)
    └── meta/            # JSON files with:
        - Control commands (steer, throttle, brake)
        - Speed and navigation command
        - Predicted trajectory (8 timesteps, x-y coordinates)
        - Predicted waypoints in world coordinates
```

## Metrics

After evaluation completes, compute metrics:

```bash
# Merge results from multiple routes
python tools/merge_route_json.py -f eval_results_diffusiondrive_*/

# Calculate multi-ability metrics
python tools/ability_benchmark.py -r merge.json

# Compute efficiency and smoothness
python tools/efficiency_smoothness_benchmark.py -f merge.json -m eval_results_diffusiondrive_*/
```

## Common Issues and Solutions

### 1. Evaluation Skips Routes

```bash
# Delete checkpoint file or set RESUME=False
rm eval_results_*/eval_*.json
```

### 2. CUDA Warning but Still Works

- Warning about CUDA version mismatch is normal
- Model still runs on GPU successfully

### 3. Segmentation Fault

- Usually occurs after ~30 seconds
- Clean CARLA processes: `bash tools/clean_carla.sh`
- Restart evaluation

### 4. No Images Saved

- Ensure SAVE_PATH is exported in script
- Check disk space availability

## Performance

- Processing speed: ~0.24x real-time with navigation commands
- GPU: RTX 6000 or similar recommended
- Memory: 32GB+ system RAM recommended
- Disk: ~1GB per route for full image saving

## Test Routes Available

- `test_routes_town10hd_only.xml` - 4 routes in Town10HD
- `test_routes_3maps.xml` - 15 routes across 3 towns
- `leaderboard/data/drivetransformer_bench2drive_dev10.xml` - 10 development routes
- `leaderboard/data/bench2drive220.xml` - Full 220 routes

## Tips

1. Always run small subset first to verify setup
2. Monitor first few minutes of evaluation for errors
3. Use `screen` or `tmux` for long evaluations
4. Save evaluation logs for debugging
5. Check trajectory predictions in meta files to verify model output

## Environment Setup (Old - kept for reference)

### 1. Set Environment Variables

```bash
# CARLA and Bench2Drive paths
export CARLA_ROOT=/path/to/your/carla  # Update this to your CARLA installation
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg

# Bench2Drive specific
export SCENARIO_RUNNER_ROOT=scenario_runner
export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export IS_BENCH2DRIVE=True

# Map location (specific to your setup)
export BENCH2DRIVE_MAP_PATH=/mnt/nvme1/dataset/Bench2Drive-Map/
```

### 2. Install Required Packages

If not already installed, install the required Python packages:

```bash
pip3 install py-trees==0.8.3 pydot
```

## File Structure

The DiffusionDrive integration consists of:

```
Bench2Drive/
├── leaderboard/team_code/
│   ├── diffusiondrive_b2d_agent.py     # Main agent for CARLA evaluation
│   ├── diffusion_model_wrapper.py      # Model wrapper without MMCV dependencies
│   ├── sensor_processor.py             # Sensor data processing
│   ├── trajectory_planner.py           # Trajectory planning logic
│   ├── pid_controller.py               # PID control for steering/throttle
│   └── planner.py                      # Route planning utilities
├── test_files/
│   ├── test_b2d.ckpt                   # Model checkpoint
│   ├── kmeans_b2d_v2_traj_20.npy       # Trajectory anchors
│   ├── pytorch_model.bin               # ResNet backbone weights
│   └── diffusiondrive_config.yaml      # Agent configuration
└── DiffusionDrive/                     # Symbolic link to DiffusionDrive repo
```

## Running Evaluation

### 1. CARLA Server Configuration

**Important**: In Docker environment, CARLA is running at hostname `carla`, not `localhost`.

```bash
# CARLA is already running at carla:2000
# To verify connection:
python3 test_carla.py
```

If you need to start CARLA manually:

```bash
${CARLA_ROOT}/CarlaUE4.sh -carla-rpc-port=2000 -graphicsadapter=0 &
```

### 2. Quick Test - Agent Initialization

Before running full evaluation, test if the agent initializes correctly:

```bash
python3 test_agent_init.py
```

Expected output:

```
✓ Agent imported successfully
✓ Model wrapper imported
✓ Model wrapper initialized successfully
```

### 3. Run Debug Evaluation (Single Route)

For testing with a single route or debugging:

```bash
# Using the evaluation script
bash run_evaluation_diffusiondrive.sh

# Or disable GPU to avoid segfault (temporary workaround)
export CUDA_VISIBLE_DEVICES=""
bash run_evaluation_diffusiondrive.sh
```

**Note**: Currently experiencing segfault issues, likely GPU-related.

### 4. Run Full Evaluation (All 220 Routes)

For complete benchmark evaluation:

```bash
# Set up evaluation parameters
export GPU_RANK=0
export TEAM_AGENT=leaderboard/team_code/diffusiondrive_b2d_agent.py
export TEAM_CONFIG=test_files/diffusiondrive_config.yaml+test_files/test_b2d.ckpt+eval_run

# Run evaluation
bash leaderboard/scripts/run_evaluation_debug.sh
```

### 5. Multi-GPU Parallel Evaluation

For faster evaluation using multiple GPUs:

```bash
# Create a custom multi-GPU script based on existing ones
# Set GPU_RANK_LIST and TASK_LIST appropriately
# Example: GPU_RANK_LIST=(0 1 2 3), TASK_LIST=(0 1 2 3 4 5 6 7)
```

## Important CARLA Configuration

### Port Configuration

- Default CARLA port: **2000** (as specified in your setup)
- Traffic Manager port: **8000** (default)
- Make sure these ports are not in use: `lsof -i:2000`

### GPU Selection

CARLA uses `-graphicsadapter` flag, not `CUDA_VISIBLE_DEVICES`:

- GPU 0: `-graphicsadapter=0`
- GPU 1: `-graphicsadapter=1` (may map differently on some systems)

### Common Issues and Solutions

1. **CARLA crashes frequently**

   ```bash
   # Clean CARLA processes between runs
   bash tools/clean_carla.sh
   ```

2. **Port conflicts**

   ```bash
   # Check port usage
   lsof -i:2000
   lsof -i:8000
   ```

3. **Missing map data**
   Ensure maps are available at `/mnt/nvme1/dataset/Bench2Drive-Map/`

## Post-Evaluation Analysis

### 1. Generate Visualization Videos

After evaluation, create debug videos from saved RGB images:

```bash
# Generate video from RGB frames
python3 tools/generate_video.py -f eval_diffusiondrive/rgb_folder/

# The script will create a video with CAN bus info overlaid on frames
```

### 2. Merge Results and Calculate Metrics

```bash
# Merge evaluation JSONs (requires exactly 220 routes)
python3 tools/merge_route_json.py -f eval_diffusiondrive/

# Calculate multi-ability metrics
python3 tools/ability_benchmark.py -r merge.json

# Calculate efficiency and smoothness
python3 tools/efficiency_smoothness_benchmark.py -f merge.json -m eval_diffusiondrive/metrics/
```

## Configuration Details

### Agent Configuration (diffusiondrive_config.yaml)

```yaml
agent:
  name: diffusiondrive_b2d
  
model:
  dataset_type: bench2drive
  num_modes: 20              # Number of trajectory modes
  num_timesteps: 8           # 4 seconds at 0.5s intervals
  n_timesteps: 50            # Truncated diffusion steps
  
sensors:
  camera:
    width: 1600
    height: 900
    fov: 100
  use_lidar: false
  
control:
  max_steering: 0.7
  max_throttle: 0.75
  max_brake: 1.0
```

### Path Configuration String Format

```
config_path+checkpoint_path+save_name
```

Example:

```
test_files/diffusiondrive_config.yaml+test_files/test_b2d.ckpt+eval_run
```

## Troubleshooting

### Current Known Issues (2025-08-27)

- **Segmentation fault during evaluation**: GPU initialization conflict with PyTorch/CARLA
- **Workaround**: Try running with `export CUDA_VISIBLE_DEVICES=""`

### Model Loading Issues

- Verify checkpoint file exists: `test_files/test_b2d.ckpt`
- Check trajectory anchors: `test_files/kmeans_b2d_v2_traj_20.npy`
- Ensure ResNet backbone is present: `test_files/pytorch_model.bin`

### CARLA Connection Issues

- **Important**: In Docker, CARLA hostname is `carla` not `localhost`
- Verify CARLA is running: `python3 test_carla.py`
- Check correct port: 2000 (not 30000)
- Clean previous instances: `bash tools/clean_carla.sh`

### Memory Issues

- Evaluation saves debug images which can consume significant disk space
- Monitor disk usage during evaluation
- Clean up after each run if needed

## Expected Output Structure

After successful evaluation:

```
eval_diffusiondrive/
├── rgb_folder/              # Debug images for video generation
├── metrics/                 # Performance metrics
├── eval_diffusiondrive.json # Evaluation results
└── logs/                    # Detailed logs
```

## Performance Notes

- Single route evaluation: ~5-10 minutes
- Full 220 routes: Several hours (depends on hardware)
- Recommended: Use multi-GPU setup for full evaluation
- GPU memory requirement: ~4-8GB per instance

## Contact and Issues

For issues specific to:

- DiffusionDrive integration: Check this documentation
- Bench2Drive framework: See main README.md
- CARLA setup: Refer to CARLA documentation

Remember to clean CARLA processes regularly and monitor system resources during evaluation!
