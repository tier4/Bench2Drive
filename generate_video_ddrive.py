#!/usr/bin/env python3
"""
Generate video from DiffusionDrive evaluation results with canbus info overlay.
"""
import cv2
import os
import numpy as np
import json
from tqdm import trange
import argparse


def create_video(images_folder, output_video, fps=10, font_scale=0.8, text_color=(0, 255, 0), text_position=(50, 50)):
    """Create video from saved images with control info overlay."""
    
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
    height, width, layers = frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print(f"Creating video: {output_video}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")
    
    for i in trange(len(images), desc="Processing frames"):
        image_file = images[i]
        image_num = os.path.splitext(image_file)[0]  # Get number from filename
        
        # Read metadata
        meta_file = os.path.join(meta_dir, f"{image_num}.json")
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                
            # Extract control info (handle nested structure)
            if 'control' in meta:
                steer = float(meta['control']['steer'])
                throttle = float(meta['control']['throttle'])
                brake = float(meta['control']['brake'])
            else:
                # Fallback to top-level
                steer = float(meta.get('steer', 0))
                throttle = float(meta.get('throttle', 0))
                brake = float(meta.get('brake', 0))
            
            speed = float(meta.get('speed', 0))
            
            # Get navigation command
            command = meta.get('command', 'UNKNOWN')
            command_id = meta.get('command_id', -1)
            
            # Convert speed from m/s to km/h
            speed_kmh = speed * 3.6
            
            # Create text overlay
            text_lines = [
                f'Command: {command}',
                f'Speed: {speed_kmh:.1f} km/h ({speed:.2f} m/s)',
                f'Steer: {steer:.3f}',
                f'Throttle: {throttle:.2f} | Brake: {brake:.2f}',
                f'Frame: {i}/{len(images)}'
            ]
        else:
            text_lines = [f'Frame: {i}/{len(images)} (no metadata)']
        
        # Read image
        img = cv2.imread(os.path.join(rgb_front_dir, image_file))
        
        # Add semi-transparent background for text
        overlay = img.copy()
        cv2.rectangle(overlay, (30, 30), (550, 30 + 30 * len(text_lines)), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
        
        # Add text overlay
        for j, line in enumerate(text_lines):
            y_pos = text_position[1] + j * 30
            cv2.putText(img, line, (text_position[0], y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)
        
        video.write(img)
    
    video.release()
    cv2.destroyAllWindows()
    print(f"Video saved: {output_video}")


def main():
    parser = argparse.ArgumentParser(description='Generate video from evaluation results')
    parser.add_argument('-f', '--folder', required=True, help='Path to the folder containing rgb_front and meta directories')
    parser.add_argument('-o', '--output', help='Output video file name (default: output_video.mp4)')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second (default: 10)')
    
    args = parser.parse_args()
    
    # Set output filename if not provided
    if args.output is None:
        folder_name = os.path.basename(os.path.normpath(args.folder))
        args.output = f"{folder_name}_video.mp4"
    
    create_video(args.folder, args.output, args.fps)


if __name__ == "__main__":
    main()