# -*- coding: utf-8 -*-
import numpy as np
import open3d as o3d
import argparse

def pcd_to_bin(pcd_path, bin_path):
    """
    Converts a .pcd point cloud file to a KITTI-compatible .bin format using the open3d library.

    Args:
        pcd_path (str): Path to the input .pcd file.
        bin_path (str): Path for the output .bin file.
    """
    try:
        # 1. Load the .pcd file using open3d
        pcd = o3d.io.read_point_cloud(pcd_path)
        print(f"Successfully loaded .pcd file: {pcd_path}")

        # 2. Get points as a NumPy array of shape (N, 3)
        points = np.asarray(pcd.points)

        # 3. Handle the intensity field
        # The KITTI .bin format requires a 4th column for intensity.
        # Open3d does not have a universal way to load arbitrary fields like 'intensity'.
        # Therefore, we will create a placeholder column of zeros.
        # For visualization and testing, this is generally sufficient.
        intensities = np.zeros((points.shape[0], 1), dtype=np.float32)
        print("Creating a placeholder intensity column with zeros.")

        # 4. Concatenate points (x, y, z) and intensities into an (N, 4) array
        kitti_points = np.hstack((points, intensities)).astype(np.float32)

        # 5. Save the array to a binary file
        kitti_points.tofile(bin_path)
        
        print("-" * 30)
        print(f"Conversion successful! Processed {len(kitti_points)} points.")
        print(f"Output file saved to: {bin_path}")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert .pcd point cloud files to KITTI .bin format using open3d.')
    parser.add_argument('input_pcd', type=str, help='Path to the input .pcd file.')
    parser.add_argument('output_bin', type=str, help='Path for the output .bin file.')
    
    args = parser.parse_args()
    
    pcd_to_bin(args.input_pcd, args.output_bin)