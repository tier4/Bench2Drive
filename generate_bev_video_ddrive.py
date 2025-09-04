#!/usr/bin/env python3
"""
Generate BEV (Bird's Eye View) video from DiffusionDrive evaluation results with predicted trajectory overlay.
"""
import cv2
import os
import numpy as np
import json
from tqdm import trange
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from PIL import Image


def create_bev_frame(meta_data, frame_idx, width=400, height=400, pixels_per_meter=10):
    """Create a BEV visualization with trajectory overlay."""
    
    # Create figure with dark background
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
    ax.set_facecolor('#1a1a1a')
    
    # Set up the coordinate system (centered on ego vehicle)
    ax.set_xlim(-width/2/pixels_per_meter, width/2/pixels_per_meter)
    ax.set_ylim(-height/2/pixels_per_meter, height/2/pixels_per_meter)
    ax.set_aspect('equal')
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Draw grid for reference
    grid_spacing = 5  # meters
    for x in np.arange(-20, 25, grid_spacing):
        ax.axvline(x, color='gray', alpha=0.2, linewidth=0.5)
    for y in np.arange(-20, 25, grid_spacing):
        ax.axhline(y, color='gray', alpha=0.2, linewidth=0.5)
    
    # Draw ego vehicle (at origin)
    ego_width = 1.8  # meters
    ego_length = 4.5  # meters
    ego_rect = FancyBboxPatch(
        (-ego_width/2, -ego_length/2), 
        ego_width, ego_length,
        boxstyle="round,pad=0.1",
        facecolor='#4CAF50', 
        edgecolor='white', 
        linewidth=2,
        alpha=0.9
    )
    ax.add_patch(ego_rect)
    
    # Draw ego vehicle direction arrow
    ax.arrow(0, 0, 0, 3, head_width=0.5, head_length=0.5, 
             fc='white', ec='white', alpha=0.8)
    
    # Get predicted trajectory if available
    if 'predicted_trajectory' in meta_data:
        trajectory = np.array(meta_data['predicted_trajectory'])
        
        # Trajectory should be in ego frame (x: forward, y: left)
        # Convert to visualization frame (x: right, y: forward)
        traj_x = -trajectory[:, 1]  # Negate for left->right
        traj_y = trajectory[:, 0]   # Forward stays forward
        
        # Draw trajectory as gradient line
        points = np.column_stack([traj_x, traj_y])
        segments = []
        colors = []
        
        for i in range(len(points) - 1):
            segments.append([points[i], points[i+1]])
            # Color gradient from green to yellow
            colors.append((1.0 - i/len(points), 1.0, 0.0, 0.8 - i/len(points)*0.4))
        
        lc = LineCollection(segments, colors=colors, linewidths=3)
        ax.add_collection(lc)
        
        # Draw waypoints
        for i, (x, y) in enumerate(points):
            circle = plt.Circle((x, y), 0.3, color='yellow', alpha=0.6 - i/len(points)*0.4)
            ax.add_patch(circle)
    
    # Add text overlay
    info_text = []
    info_text.append(f"Frame: {frame_idx}")
    
    if 'command' in meta_data:
        info_text.append(f"Command: {meta_data['command']}")
    
    if 'speed' in meta_data:
        speed_mps = meta_data['speed']
        speed_kmh = speed_mps * 3.6
        info_text.append(f"Speed: {speed_kmh:.1f} km/h")
    
    if 'control' in meta_data:
        control = meta_data['control']
        info_text.append(f"Steer: {control['steer']:.3f}")
        info_text.append(f"Throttle: {control['throttle']:.2f}")
        info_text.append(f"Brake: {control['brake']:.2f}")
    
    # Add text box
    text_str = '\n'.join(info_text)
    props = dict(boxstyle='round', facecolor='black', alpha=0.7)
    ax.text(-19, 19, text_str, fontsize=10, color='white',
            verticalalignment='top', bbox=props)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#4CAF50', label='Ego Vehicle'),
        mpatches.Patch(color='yellow', label='Predicted Path'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', 
             facecolor='black', edgecolor='white', labelcolor='white')
    
    # Add title
    ax.set_title('Bird\'s Eye View - Trajectory Prediction', 
                color='white', fontsize=14, pad=20)
    
    # Convert to image
    fig.canvas.draw()
    # Use buffer_rgba for compatibility
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    # Convert RGBA to RGB
    buf = buf[:, :, :3]
    
    plt.close(fig)
    
    return buf


def create_combined_video(images_folder, output_video, fps=10):
    """Create video combining front camera and BEV visualization."""
    
    rgb_front_dir = os.path.join(images_folder, 'rgb_front')
    meta_dir = os.path.join(images_folder, 'meta')
    
    # Get all images
    images = [img for img in os.listdir(rgb_front_dir) if img.endswith((".jpg", ".png"))]
    images.sort()
    
    if not images:
        print(f"No images found in {rgb_front_dir}")
        return
    
    print(f"Found {len(images)} images")
    
    # Read first frame to get dimensions
    frame = cv2.imread(os.path.join(rgb_front_dir, images[0]))
    h_rgb, w_rgb, _ = frame.shape
    
    # Create first BEV to get dimensions
    meta_file = os.path.join(meta_dir, f"{os.path.splitext(images[0])[0]}.json")
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            meta = json.load(f)
    else:
        meta = {}
    
    bev_frame = create_bev_frame(meta, 0)
    h_bev, w_bev, _ = bev_frame.shape
    
    # Resize BEV to match RGB height
    scale = h_rgb / h_bev
    w_bev_resized = int(w_bev * scale)
    
    # Combined frame dimensions
    combined_width = w_rgb + w_bev_resized
    combined_height = h_rgb
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (combined_width, combined_height))
    
    print(f"Creating video: {output_video}")
    print(f"Resolution: {combined_width}x{combined_height}, FPS: {fps}")
    
    for i in trange(len(images), desc="Processing frames"):
        image_file = images[i]
        image_num = os.path.splitext(image_file)[0]
        
        # Read RGB frame
        rgb_frame = cv2.imread(os.path.join(rgb_front_dir, image_file))
        
        # Read metadata and add predicted trajectory
        meta_file = os.path.join(meta_dir, f"{image_num}.json")
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            
            # Use ACTUAL predicted trajectory from model - NO MOCK DATA!
        else:
            meta = {}  # NO MOCK DATA - if no metadata, show no trajectory
        
        # Create BEV frame
        bev_frame = create_bev_frame(meta, i)
        
        # Resize BEV to match RGB height
        bev_frame_resized = cv2.resize(bev_frame, (w_bev_resized, h_rgb))
        
        # Convert BEV from RGB to BGR for OpenCV
        bev_frame_bgr = cv2.cvtColor(bev_frame_resized, cv2.COLOR_RGB2BGR)
        
        # Combine frames horizontally
        combined_frame = np.hstack([rgb_frame, bev_frame_bgr])
        
        # Add divider line
        cv2.line(combined_frame, (w_rgb, 0), (w_rgb, combined_height), (255, 255, 255), 2)
        
        video.write(combined_frame)
    
    video.release()
    cv2.destroyAllWindows()
    print(f"Video saved: {output_video}")


def main():
    parser = argparse.ArgumentParser(description='Generate BEV video from evaluation results')
    parser.add_argument('-f', '--folder', required=True, 
                       help='Path to the folder containing rgb_front and meta directories')
    parser.add_argument('-o', '--output', 
                       help='Output video file name (default: output_bev_video.mp4)')
    parser.add_argument('--fps', type=int, default=10, 
                       help='Frames per second (default: 10)')
    parser.add_argument('--bev-only', action='store_true',
                       help='Generate only BEV visualization without front camera')
    
    args = parser.parse_args()
    
    # Set output filename if not provided
    if args.output is None:
        folder_name = os.path.basename(os.path.normpath(args.folder))
        args.output = f"{folder_name}_bev_video.mp4"
    
    create_combined_video(args.folder, args.output, args.fps)


if __name__ == "__main__":
    main()