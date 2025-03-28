"""
Matrix Sampling Script
Purpose:
    This script divides the input matrix into four equal parts and outputs the top left portion

Input:
    - Dataset directory containing sparse matrix files in `.mtx` format (COO format).

Output:
    - Smaller matrix

Example Usage:
    Run the script from the terminal:
        $ python3 /home/OptixSDK/Tool/PythonTool/src/matrixSampling.py /home/OptixSDK/Tool/PythonTool/data/dw256B.mtx /home/OptixSDK/Tool/PythonTool/data/dw256B_small.mtx
        $ python3 /home/OptixSDK/Tool/PythonTool/src/matrixSampling.py /home/trace/sparse_matrices/suitSparse/all/webbase-1M/webbase-1M.mtx ./temp.mtx
"""


from scipy.io import mmread, mmwrite
import numpy as np
import sys

"""
Extract the top-left quarter of a matrix.
"""
def sample_top_left(input_file, output_file):
    # Read the input matrix
    matrix = mmread(input_file).tocsc()  # Ensure the matrix is in sparse CSC format
    
    # Get the matrix dimensions
    rows, cols = matrix.shape
    
    # Determine the midpoint for division
    mid_row = rows // 2
    mid_col = cols // 2
    
    # Extract the top-left portion
    top_left_matrix = matrix[:mid_row, :mid_col]
    
    # Write the top-left portion to the output file
    mmwrite(output_file, top_left_matrix)
    print(f"Top-left portion saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file.mtx> <output_file.mtx>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    sample_top_left(input_file, output_file)
