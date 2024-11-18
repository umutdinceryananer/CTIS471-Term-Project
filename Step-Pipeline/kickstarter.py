import subprocess


# This code was written to sequentially trigger manual pipeline elements as the process ends.

python_files = [
    'Step-Pipeline/1_metric_manipulation.py',
    'Step-Pipeline/2_anomaly_removal.py',  
    'Step-Pipeline/3_dataset_prep.py'   
]

success_count = 0  

for file in python_files:
    print(f"Running {file}...")
    result = subprocess.run(['python', file], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"{file} completed successfully.")
        success_count += 1  
    else:
        print(f"Error occurred while running {file}:")
        print(result.stderr)

if success_count == len(python_files):
    print("Small and Large Datasets Created.")
