"""
Dataset Profiling Script

"""

import os
import subprocess
import shutil
import sys
import argparse
from scipy.io import mmread, mmwrite
import numpy as np


# Specify path
DATA_DIR = "/home/RTSpMSpM/optixSpMSpM/data/"
PROF_DIR = "/home/RTSpMSpM"
TEMP_LOG = "/home/RTSpMSpM/scripts/temp.txt"
MATRIX_SAMPLING_SCRIPT = "/home/RTSpMSpM/scripts/matrixSampling.py"
# Dataset dict, names as keys and whether they are squared mat as values
dataset = {
    "p2p-Gnutella31"   : True,
    "roadNet-CA"       : True, 
    "webbase-1M"       : True,    
    "mario002"         : True,    
    "web-Google"       : True,  
    "scircuit"         : True,    
    "amazon0312"       : True, 
    "ca-CondMat"       : True, 
    "email-Enron"      : True,
    "wiki-Vote"        : True,
    "cage12"           : True,    
    "2cubes_sphere"    : True,    
    "offshore"         : True,    
    "cop20k_A"         : True,    
    "filter3D"         : True,    
    "poisson3Da"       : True      
}
for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".mtx"):
            dataset_name = os.path.splitext(file)[0]  # Get filename without extension
            dataset[dataset_name] = True  # Assume all matrices are squared for now
print(f"Discovered datasets: {list(dataset.keys())}")


# Dictionary of programs with program names as keys and binary paths as values
program_list = {
    "optixSpMSpM"     : "/home/RTSpMSpM/optixSpMSpM/build/bin/optixSpMSpM",
    "cuSparse"        : "/home/RTSpMSpM/cuSparse/src/cuSparse"
    # "intelMKL"        : "/home/OptixSDK/IntelMKL/mkl_spmspm/bin/sparseMKL"
}
num_run = 1

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

    if mid_row < 1000:
        print(f"Error: matrix too small")
        return False
        # sys.exit(f"Error: matrix too small")
    
    print(f"Size is now {mid_row}")

    # Extract the top-left portion
    top_left_matrix = matrix[:mid_row, :mid_col]
    
    if top_left_matrix.nnz == 0:
        # sys.exit("Error: matrix has no elements")
        print("Error: matrix has no elements")
        return False
    
    # Write the top-left portion to the output file
    mmwrite(output_file, top_left_matrix)
    print(f"Top-left portion saved to {output_file}")
    return True


def run_matrix_sampling(input_matrix, output_matrix):
    """
    Reduces the matrix size using the matrixSampling script.
    """
    try:
        print(f"Running matrix sampling: {input_matrix} -> {output_matrix}")
        # Call the sample_top_left function directly
        if sample_top_left(input_matrix, output_matrix):
            print(f"Matrix sampling completed successfully: {input_matrix} -> {output_matrix}")
            return True
        else:
            return False
    except Exception as e:
        print(f"Error during matrix sampling: {e}")
        sys.exit(1)  # Exit the script if sampling fails



def main():
    all_failed_data_files = set()
    optix_not_failed_data = {}

    # Open the file in append mode
    PROF_FILE_PATH = os.path.join(PROF_DIR, "result.csv")
    with open(PROF_FILE_PATH, "a") as f:
        f.write("Software, DataSet, Scenario, Runtime(ms)\n")

    # Iterate over each data file in the DATA_DIR
    for data_file in dataset:
        # Construct the full path to the data file
        data_file_path = os.path.join(DATA_DIR, f"{data_file}/{data_file}.mtx")

        # Check if it's a file (and check we need to )
        if not os.path.isfile(data_file_path):
            data_file_path = os.path.join(DATA_DIR, f"{data_file}.mtx")
            if not os.path.isfile(data_file_path):
                sys.exit(f"Error: data file not found {data_file_path}")

        # If file is not squared, we need to use its transpose as the 2nd input
        is_squared = dataset[data_file]
        if not is_squared:
            transpose_file_path = os.path.join(DATA_DIR, f"{data_file}/{data_file}_transpose.mtx")
            if not os.path.isfile(transpose_file_path):
                sys.exit(f"Error: data file not found {transpose_file_path}")
        
        # Create directory 
        result_path = os.path.join(prof_path, "result/")
        os.makedirs(result_path, exist_ok=create_new_dir)

        success = 0
        data_path = data_file_path
        # If we tested this with the edited dataset
        if os.path.isfile(os.path.join(result_path, f"{data_file}_small.mtx")):
            data_path = os.path.join(result_path, f"{data_file}_small.mtx")

        while not success:
            failed = False
            print("Working...")
            out_buff = ""

            # Iterate over each program
            for program_name, program_path in program_list.items():
                program_result_path = os.path.join(result_path, f"{program_name}.mtx")
                program_prof_path = os.path.join(prof_path, f"{program_name}/")
                os.makedirs(program_prof_path, exist_ok=create_new_dir)

                # Run the program with the data file as an argument
                print(f"Running {program_name} ({program_path}) with input {data_path}")
                try:
                    for i in range(num_run):
                        out_matrix_path = os.path.join(result_path, f"{program_name}.mtx")
                        out_buff += program_name + ", " + data_file + ", " + f"run{i}, "
                        # Run program
                        # if program_name == "intelMKL":
                        #     command = f"bash -c 'source /opt/intel/oneapi/setvars.sh && {program_path} -m1 {data_path} -m2 {data_path} -o {out_matrix_path}'"
                        #     subprocess.run(command, shell=True, check=True, text=True)
                        #     continue
                        if is_squared:
                            # IMPORTANT: optix needs relative path
                            if "optix" in program_name:
                                rel_data_path = os.path.relpath(data_path, start=os.path.dirname(program_path))
                                subprocess.run(
                                    [program_path, "-m1", f"{rel_data_path}", "-m2", f"{rel_data_path}", "-o", out_matrix_path, "-l", TEMP_LOG],
                                    check=True
                                )
                            else:
                                subprocess.run(
                                    [program_path, "-m1", f"{data_path}", "-m2", f"{data_path}", "-o", out_matrix_path, "-l", TEMP_LOG],
                                    check=True
                                )
                        else:
                            sys.exit(f"Error: unable to run not squared matrix TODO")
                        with open(TEMP_LOG, "r") as file:
                            content = file.read()
                            out_buff += content + "\n"
                except subprocess.CalledProcessError as e:
                    print(f"Error running {program_name} with {data_file_path} : {e}\n")
                    # If optix failed, reduce size
                    # If optix have not failed but others did, then record
                    if "optix" not in program_name and not failed:
                        if program_name not in optix_not_failed_data:
                            optix_not_failed_data[program_name] = set()
                        # Add the failing data file to the program's set
                        optix_not_failed_data[program_name].add(data_file)
                    else:
                        failed = True

            # Reduce the size if optix failed
            if failed:
                # Perform matrix sampling and restart
                # print("Reducing matrix size due to failure.")
                sampled_file_path = os.path.join(result_path, f"{data_file}_small.mtx")
                if not run_matrix_sampling(data_path, sampled_file_path):
                    all_failed_data_files.add(data_file)
                    break
                data_path = sampled_file_path  # Use the reduced matrix for further runs
            else:
                success = 1
                with open(PROF_FILE_PATH, "a") as output_file:
                    output_file.write(out_buff)
                print(f"All programs succeeded for {data_file}")
    
    # print("Data files that caused error (no elements after sampling):", all_failed_data_files)

    if optix_not_failed_data:
        # print("\nPrograms that failed while OptiX succeeded:")
        for program, failed_files in optix_not_failed_data.items():
            print(f"  {program}: {', '.join(failed_files)}")
    else:
        print("\nAll programs succeeded without failures while OptiX was successful.")


if __name__ == "__main__":
    main()
