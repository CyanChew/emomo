import os
import shutil
import argparse

def collect_npz_files(data_paths, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Determine the starting index
    existing_files = [f for f in os.listdir(output_folder) if f.endswith(".npz")]
    existing_indices = [int(f.replace("demo", "").replace(".npz", "")) for f in existing_files if f.startswith("demo") and f.replace("demo", "").replace(".npz", "").isdigit()]
    next_index = max(existing_indices, default=-1) + 1
    
    for data_path in data_paths:
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} does not exist. Skipping.")
            continue
        
        npz_files = sorted([f for f in os.listdir(data_path) if f.endswith(".npz")])
        
        for file_name in npz_files:
            source_file = os.path.join(data_path, file_name)
            destination_file = os.path.join(output_folder, f"demo{next_index:05d}.npz")
            
            shutil.copy2(source_file, destination_file)
            print(f"Copied {source_file} to {destination_file}")
            next_index += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect .npz files from multiple directories into a single output folder.")
    parser.add_argument("data_paths", nargs='+', help="List of input directories containing .npz files.")
    parser.add_argument("output_folder", help="Directory where all .npz files will be collected.")
    
    args = parser.parse_args()
    collect_npz_files(args.data_paths, args.output_folder)
