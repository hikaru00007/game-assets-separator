import subprocess
import sys
import os
import cv2
import numpy as np
from pathlib import Path

# 1. Automatic Library Initialization
def install_requirements():
    required = {'opencv-python', 'numpy'}
    installed = {pkg.split('==')[0] for pkg in subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode().split('\n')}
    missing = required - installed
    if missing:
        print(f"Installing missing libraries: {missing}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])

install_requirements()

# 2. Batch Processing
base_dir = Path(__file__).resolve().parent

for file_path in base_dir.glob("*.png"):
    # Skip already extracted folders or system files
    if file_path.stem == "extracted_assets" or file_path.is_dir():
        continue
        
    print(f"Processing: {file_path.name}")
    
    # Create subdir ./filename/
    output_dir = base_dir / file_path.stem
    output_dir.mkdir(exist_ok=True)
    
    img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if img is None: continue
    
    # Masking: Detect solid objects (ignore transparent background)
    alpha = img[:, :, 3]
    _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    
    # Connected Components (8-connectivity)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Export
    count = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 50:
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            roi = img[y:y+h, x:x+w]
            cv2.imwrite(str(output_dir / f"{file_path.stem}_{count}.png"), roi)
            count += 1

print("Done. All assets separated into their respective folders.")