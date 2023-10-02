from nltk import corpus
import time
from logAnalyzier import *
import pandas as pd
from CheckPoints import CheckPointsEnum
import csv

def Transform_To_CSV(data, csv_filename="output.csv"):
    # Convert the list of dictionaries to list of lists for csv writer
    rows = []
    for item in data:
        for file_name, checkpoints in item.items():
            row = [file_name]
            row.extend([checkpoints[f'CP{i}'] for i in range(1, 7)])
            row.append(sum(row[1:7]))
            rows.append(row)

    # Write to CSV
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['File Name', 'Checkpoint 1', 'Checkpoint 2', 'Checkpoint 3', 'Checkpoint 4', 'Checkpoint 5', 'Checkpoint 6', 'Total Score'])
        # Write rows
        writer.writerows(rows)

if __name__ == '__main__':


    log_analyzer = LogAnalyzer()
    log_file_path = ["sample1_logs.txt","test.txt","test2.txt"]

    start_time = time.time()
    res = log_analyzer.process_multiple_files(log_file_path)
    end_time = time.time()
    print(f"Time taken to process {log_file_path}: {end_time - start_time} seconds")
    print(res)
    Transform_To_CSV(res)

