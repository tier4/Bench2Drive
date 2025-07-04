#!/home/binwang/miniforge3/bin/python3
"""
VAD Visualization with Metadata Display
Shows existing rendered images with comprehensive metadata overlay
No trajectory projections - uses pre-rendered BEV with trajectories
"""
import cv2
import os
import numpy as np
import json
import argparse
from tqdm import tqdm
import glob
from multiprocessing import Pool, cpu_count
from functools import partial

# Global Constants
COMMAND_NAMES = {
    0: "STRAIGHT",
    1: "TURN_LEFT", 
    2: "TURN_RIGHT",
    3: "LANE_FOLLOW",
    4: "CHANGE_LEFT",
    5: "CHANGE_RIGHT"
}


def safe_load_image(img_path, target_size=None):
    """Safely load an image with error handling."""
    if not os.path.exists(img_path):
        print(f"Warning: Image not found: {img_path}")
        if target_size:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        return np.zeros((300, 533, 3), dtype=np.uint8)
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not decode image: {img_path}")
        if target_size:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        return np.zeros((300, 533, 3), dtype=np.uint8)
    
    if target_size and img.shape[:2][::-1] != target_size:
        img = cv2.resize(img, target_size)
    
    return img


def create_control_panel(meta_data, panel_width=533, panel_height=300):
    """Create control panel with all metadata information."""
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    
    # Title
    y_offset = 30
    x_offset = 20
    cv2.putText(panel, "CONTROL INFORMATION", (x_offset, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Driving Command (1st line)
    y_offset += 35
    command = meta_data.get('command', -1)
    cmd_name = COMMAND_NAMES.get(command, "UNKNOWN")
    cv2.putText(panel, f"Command: {cmd_name}", 
               (x_offset, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 1)
    
    # Target Speed (2nd line)
    y_offset += 30
    desired_speed = meta_data.get('desired_speed', 0) * 3.6  # m/s to km/h
    cv2.putText(panel, f"Target Speed: {desired_speed:.1f} km/h", 
               (x_offset, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
    
    # Current Speed (3rd line)
    y_offset += 30
    speed = meta_data.get('speed', 0) * 3.6  # m/s to km/h
    cv2.putText(panel, f"Speed: {speed:.1f} km/h", 
               (x_offset, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    
    # Steering
    y_offset += 30
    steer = meta_data.get('steer', 0)
    cv2.putText(panel, f"Steer: {steer:+.3f}", 
               (x_offset, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    
    # Throttle
    y_offset += 30
    throttle = meta_data.get('throttle', 0)
    cv2.putText(panel, f"Throttle: {throttle:.2f}", 
               (x_offset, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    
    # Brake
    y_offset += 30
    brake = meta_data.get('brake', 0)
    cv2.putText(panel, f"Brake: {brake:.2f}", 
               (x_offset, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
               (255, 0, 0) if brake > 0 else (0, 255, 0), 1)
    
    # Agent Type
    y_offset += 30
    agent = meta_data.get('agent', 'unknown')
    cv2.putText(panel, f"Agent: {agent}", 
               (x_offset, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    return panel


def create_info_panel(dataset_name, frame_idx, total_frames, fps=10, panel_width=533, panel_height=300):
    """Create information panel with dataset and frame info."""
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    
    # Title
    y_offset = 30
    x_offset = 20
    cv2.putText(panel, "DATASET INFORMATION", (x_offset, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Parse dataset name for display
    parts = dataset_name.split('_')
    if len(parts) >= 3:
        scenario = parts[1]  # RouteScenario
        scenario_id = parts[2]  # ID number
        town = parts[4] if len(parts) > 4 else "Unknown"
        task = '_'.join(parts[5:-4]) if len(parts) > 9 else "Unknown"
    else:
        scenario = "Unknown"
        scenario_id = "0"
        town = "Unknown"
        task = "Unknown"
    
    # Scenario Type
    y_offset += 35
    cv2.putText(panel, f"Scenario: {scenario} {scenario_id}", 
               (x_offset, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Town
    y_offset += 30
    cv2.putText(panel, f"Town: {town}", 
               (x_offset, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Task Name
    y_offset += 30
    # Truncate long task names
    if len(task) > 25:
        task = task[:22] + "..."
    cv2.putText(panel, f"Task: {task}", 
               (x_offset, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Frame Information
    y_offset += 40
    cv2.putText(panel, f"Frame: {frame_idx:04d} / {total_frames:04d}", 
               (x_offset, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    
    # Time
    y_offset += 30
    time_sec = frame_idx / fps
    cv2.putText(panel, f"Time: {time_sec:.1f}s", 
               (x_offset, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    
    # Progress Bar
    y_offset += 40
    bar_width = panel_width - 2 * x_offset
    bar_height = 20
    progress = frame_idx / max(total_frames - 1, 1)
    
    # Draw progress bar background
    cv2.rectangle(panel, (x_offset, y_offset), 
                 (x_offset + bar_width, y_offset + bar_height), 
                 (50, 50, 50), -1)
    
    # Draw progress bar fill
    fill_width = int(bar_width * progress)
    if fill_width > 0:
        cv2.rectangle(panel, (x_offset, y_offset), 
                     (x_offset + fill_width, y_offset + bar_height), 
                     (0, 255, 0), -1)
    
    # Progress percentage
    cv2.putText(panel, f"{progress*100:.0f}%", 
               (x_offset + bar_width//2 - 20, y_offset + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return panel


def process_single_dataset(input_dir, output_path, fps=10):
    """Process a single dataset and create visualization video."""
    
    # Get list of images from any camera folder to determine frame count
    front_dir = os.path.join(input_dir, 'rgb_front')
    if not os.path.exists(front_dir):
        print(f"Skipping {input_dir}: No rgb_front directory")
        return False
    
    images = sorted([img for img in os.listdir(front_dir) if img.endswith(".png")])
    if not images:
        print(f"Skipping {input_dir}: No images found")
        return False
    
    total_frames = len(images)
    dataset_name = os.path.basename(input_dir)
    print(f"Processing {dataset_name} with {total_frames} frames...")
    
    # Layout configuration
    view_w, view_h = 533, 300
    out_w = view_w * 3
    out_h = view_h * 3
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    
    if not video.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return False
    
    # Process each frame
    for i in tqdm(range(total_frames), desc="Frames", leave=False):
        # Create canvas
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        
        # Camera layout (BEV and back exchanged)
        layout = {
            'rgb_front_left': (0, 0),
            'rgb_front': (view_w, 0),
            'rgb_front_right': (view_w * 2, 0),
            'rgb_back_left': (0, view_h),
            'rgb_back': (view_w, view_h),  # Back moved to center middle
            'rgb_back_right': (view_w * 2, view_h),
            'bev': (view_w, view_h * 2)  # BEV moved to bottom center
        }
        
        # Load and place images
        for view_name, (x, y) in layout.items():
            img_path = os.path.join(input_dir, view_name, f'{i:04d}.png')
            img = safe_load_image(img_path, (view_w, view_h))
            canvas[y:y+view_h, x:x+view_w] = img
            
            # Add view label
            label = view_name.replace('rgb_', '').upper()
            cv2.putText(canvas, label, (x + 10, y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Load metadata
        meta_file = os.path.join(input_dir, 'meta', f'{i:04d}.json')
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                meta = json.load(f)
        else:
            meta = {}
        
        # Create and place control panel (bottom left)
        control_panel = create_control_panel(meta)
        canvas[view_h * 2:view_h * 3, 0:view_w] = control_panel
        
        # Create and place info panel (bottom right)
        info_panel = create_info_panel(dataset_name, i, total_frames, fps)
        canvas[view_h * 2:view_h * 3, view_w * 2:view_w * 3] = info_panel
        
        # Add main title bar at the top
        title_height = 40
        title_bar = np.zeros((title_height, out_w, 3), dtype=np.uint8)
        cv2.rectangle(title_bar, (0, 0), (out_w, title_height), (40, 40, 40), -1)
        
        # Add title text
        title_text = f"VAD Visualization - {dataset_name}"
        if len(title_text) > 100:
            title_text = title_text[:97] + "..."
        cv2.putText(title_bar, title_text, 
                   (20, 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Create final frame with title bar
        final_frame = np.vstack([title_bar, canvas])
        
        # Resize to original dimensions
        final_frame = cv2.resize(final_frame, (out_w, out_h))
        
        video.write(final_frame)
    
    video.release()
    print(f"  Saved: {output_path}")
    return True


def process_dataset_wrapper(args):
    """Wrapper function for multiprocessing. Returns (success, dataset_name)."""
    dataset_path, output_dir, fps = args
    dataset_name = os.path.basename(dataset_path)
    output_path = os.path.join(output_dir, f"{dataset_name}.mp4")
    
    try:
        success = process_single_dataset(dataset_path, output_path, fps)
        return (success, dataset_name, None)
    except Exception as e:
        return (False, dataset_name, str(e))


def batch_process_datasets(base_dir, output_dir, fps=10, num_workers=None):
    """Process all datasets in a directory using multiprocessing."""
    
    # Find all subdirectories with VAD data structure
    datasets = []
    for item in sorted(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if it has VAD visualization structure
            required_dirs = ['rgb_front', 'bev', 'meta']
            if all(os.path.exists(os.path.join(item_path, d)) for d in required_dirs):
                datasets.append(item_path)
    
    if not datasets:
        print(f"No valid VAD visualization datasets found in {base_dir}")
        return
    
    print(f"Found {len(datasets)} datasets to process")
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), len(datasets))
    else:
        num_workers = min(num_workers, len(datasets))
    
    print(f"Using {num_workers} worker processes")
    
    # Prepare arguments for multiprocessing
    process_args = [(dataset_path, output_dir, fps) for dataset_path in datasets]
    
    # Process datasets in parallel
    success_count = 0
    failed_datasets = []
    
    with Pool(processes=num_workers) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(process_dataset_wrapper, process_args),
            total=len(datasets),
            desc="Processing datasets"
        ))
    
    # Collect results
    for success, dataset_name, error in results:
        if success:
            success_count += 1
        else:
            failed_datasets.append((dataset_name, error))
    
    print(f"\nCompleted: {success_count}/{len(datasets)} datasets processed successfully")
    
    if failed_datasets:
        print(f"\nFailed datasets ({len(failed_datasets)}):")
        for name, error in failed_datasets:
            if error:
                print(f"  - {name}: {error}")
            else:
                print(f"  - {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate VAD visualization videos with metadata display'
    )
    parser.add_argument('input_path', 
                       help='Path to single dataset or directory containing multiple datasets')
    parser.add_argument('-o', '--output', 
                       help='Output path (default: ./vis_with_traj/)')
    parser.add_argument('--fps', type=int, default=10, 
                       help='Frames per second (default: 10)')
    parser.add_argument('--batch', action='store_true', 
                       help='Process all datasets in input directory')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes for batch mode (default: auto)')
    
    args = parser.parse_args()
    
    # Set default output directory
    if not args.output:
        args.output = './vis_with_traj'
    
    if args.batch:
        # Batch processing mode
        if not os.path.isdir(args.input_path):
            print("Error: Batch mode requires input_path to be a directory")
            exit(1)
        
        print(f"Batch processing datasets in: {args.input_path}")
        print(f"Output directory: {args.output}")
        
        batch_process_datasets(args.input_path, args.output, args.fps, args.workers)
    
    else:
        # Single dataset mode
        if not os.path.isdir(args.input_path):
            print("Error: Input path must be a valid dataset directory")
            exit(1)
        
        # Generate output path
        if os.path.isdir(args.output):
            # Output is a directory, generate filename
            dataset_name = os.path.basename(os.path.normpath(args.input_path))
            output_path = os.path.join(args.output, f"{dataset_name}.mp4")
        else:
            # Output is a file path
            output_path = args.output
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Processing: {args.input_path}")
        print(f"Output: {output_path}")
        
        process_single_dataset(args.input_path, output_path, args.fps)