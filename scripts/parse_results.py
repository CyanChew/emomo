import argparse
import os
import re

def count_cumulative_successes(log_path):
    cumulative_successes = 0
    last_success = 0
    stop_seed = "s100018"

    with open(log_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if stop_seed in line:
            break
        match = re.search(r"# Success: (\d+)", line)
        if match:
            current_success = int(match.group(1))
            if current_success >= last_success:
                cumulative_successes += (current_success - last_success)
            else:
                # success counter reset, treat current_success as starting new block
                cumulative_successes += current_success
            last_success = current_success

    print(f"Cumulative successes: {cumulative_successes}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exps_dir", type=str, help="Path to eval.log file")
    args = parser.parse_args()

    count_cumulative_successes(os.path.join(args.exps_dir, 'eval.log'))

