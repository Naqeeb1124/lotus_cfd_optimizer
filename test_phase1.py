import sys
import os
import logging
import traceback

# Add project root AND src directory to Python Path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "cfd_opt", "py_files"))

# Import target objective function
from src.cfd import objective_function

# Unbuffer stdout so logs flush immediately to file
sys.stdout.reconfigure(line_buffering=True)

def run_test():
    shelf_quantity = 2  # Change to 20 for full-scale test
    print("--- Starting Phase 1 End-to-End Test ---", flush=True)
    print(f"Testing Parameters: lip_w=40.0mm, shelf_spacing=90.0mm, "
          f"shelves={shelf_quantity}", flush=True)
    
    try:
        final_cov = objective_function([40.0, 90.0], shelf_quantity=shelf_quantity)
        print(f"\n--- Test Successful! ---\nFinal CoV: {final_cov}", flush=True)
        with open("status.txt", "w") as f:
            f.write("SUCCESS")
    except Exception as e:
        print(f"\n--- Test Failed! ---\nError: {e}", flush=True)
        with open("error_traceback.log", "w") as f:
            traceback.print_exc(file=f)
        with open("status.txt", "w") as f:
            f.write("FAILED")
        raise e

if __name__ == "__main__":
    run_test()