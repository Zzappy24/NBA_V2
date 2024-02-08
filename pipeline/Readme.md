# README.md

## Setup Instructions

1. Install the required dependencies using the following command:

   ```bash
   pip install -r requirements.txt
   ```
   or

   ```bash
   conda create --name NBAMlOPS --file requirementsCondaEnv.txt
   ```


2. Open a terminal to execute the script.

3. Run the watcher script:

   ```bash
   python watcher.py
   ```

## Script Overview

- The watcher script monitors the `dataTemp/rawTemp` folder for new file additions every second.

- When a new file is detected, the main script (`main.py`) is automatically executed.

- If the pipeline runs successfully, you can find the processed files in the following directories within the `data` folder:

  - **Raw:** Contains the original/raw file.
  
  - **Curated:** Holds the cleaned version of the file.
  
  - **Training:** Consists of the file adapted to the model's requirements.
  
  - **Model:** Contains the model that is either newly trained with the data or retrained from the last model if available.

You can also check if the modification you have made can still run the pipeline with pytest :

   ```bash
   pytest
   ```
