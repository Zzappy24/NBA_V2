from path.path import CURATED_DATA_DIR_TEMP, TRANFORMED_DATA_DIR_TEMP, MODEL_TEMP_DIR, RAW_DATA_DIR_TEMP, RAW_DATA_DIR
import unittest
import subprocess
import sys
import os
import multiprocessing
import threading
import time
import shutil
from pathlib import Path
from DataCleaning.DataCleaningFunctions import main_cleaning
from DataTraining.DataTrainingFunctions import main_training
from DataTransforming.DataTransformingFunctions import main_transforming
import signal



import pandas as pd

class TestUnittest(unittest.TestCase):
    def run_subprocess(self, command, completion_event):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        completion_event.set()  
        return process.returncode, stdout, stderr

    def move_file_thread(self, completion_event):
        shutil.copy("tests/test.csv", "dataTemp/raw_temp")
        completion_event.set()

    def move_file(self):
        shutil.copy("tests/test.csv", "dataTemp/raw_temp")


    def delete_file(self, path):
        try:
            os.remove([f for f in Path(path).iterdir() if f.is_file()][0])
        except Exception:
            pass


    def test_cleaning(self):
        self.move_file()
        self.assertIsInstance(main_cleaning(), pd.DataFrame)
        self.delete_file(RAW_DATA_DIR_TEMP)


    def test_transforming(self):
        self.move_file()
        df_cleaned = main_cleaning()
        df_cleaned.to_csv(f'{CURATED_DATA_DIR_TEMP}/df_cleaned.csv', index=False)
        self.assertIsInstance(main_transforming(), pd.DataFrame)
        self.delete_file(RAW_DATA_DIR_TEMP)
        self.delete_file(CURATED_DATA_DIR_TEMP)



    def test_training(self):
        self.move_file()
        df_cleaned = main_cleaning()
        df_cleaned.to_csv(f'{CURATED_DATA_DIR_TEMP}/df_cleaned.csv', index=False)
        df_transformed = main_transforming()
        df_transformed.to_csv(f"{TRANFORMED_DATA_DIR_TEMP}/df_transformed.csv", index=False)
        rmse, cm, f1, accuracy, precision, recall, roc_auc, log_loss_val, mae, r_squared, model = main_training()
        self.delete_file(RAW_DATA_DIR_TEMP)
        self.delete_file(CURATED_DATA_DIR_TEMP)
        self.delete_file(TRANFORMED_DATA_DIR_TEMP)


    def test_EndToEnd(self):
        command1 = ["python", "watcher.py"]

        # Create an event to signal the completion of the second thread
        completion_event = threading.Event()

        # Start the first thread for the subprocess
        thread1 = threading.Thread(target=self.run_subprocess, args=(command1, completion_event))
        thread1.start()

        # Start the second thread for the file operation
        thread2 = threading.Thread(target=self.move_file_thread, args=(completion_event,))
        thread2.start()

        # Wait for the second thread to finish
        completion_event.wait()

        # Wait for 10 seconds after the second thread finishes
        time.sleep(10)

        # Wait for the first thread to finish
        thread1.terminate()

        thread1.join()

        


if __name__ == '__main__':
    unittest.main()
