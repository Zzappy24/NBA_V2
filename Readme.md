
# Use it with docker-compose

1. Install docker
   https://www.docker.com/products/docker-desktop/ 

   If you have a mac with ARM ships :
   go in settings (shown at top right), 
   features in development, 
   Use Rosetta for x86/amd64 emulation on Apple Silicon

2. download the docker-compose file 
   or git clone the all project if you want

3. Open a terminal and run :
    cd {path_to_the_docker-compose_folder}

   choose a path to have your data where you want, no need to create it, it will be done automatically:
   for exemple /Users/{your_name_user}/Desktop/test_docker

   and then run :

   ```bash
   MAIN_PATH=/Users/{your_name_user}/Desktop/test_docker docker-compose up
   ```

4. Open the navigator of your choice and write http://localhost:8501/

5. now you have access to the streamlit app, and have a visualisation of all the pipeline but with a better interface 🔥🔥

6. download the CSV files from https://drive.google.com/drive/folders/1L-cMcsmksmkGcWBqszUhzIDM6u7RLlsy
that contains the data given by https://www.basketball-reference.com/leagues/ ❤️


7. Upload a NBA CSV file, OK the pipeline is running now

8. you have now the first results of the pipeline. 
   Go to the page : Choose a pipeline version
   each run is versionned by timestamp

9. You can see the data saved for each steap and you have access to a csv line with some interestings metrics (maybe I will add more in the future)

10. you also have access to logs, allowing you to see if an error appears during the training
indeed, we have a protected pipeline, if an error appears and make the pipeline crash, no data will be saved.

11. ## DEMONSTATION
![pipeline_video-ezgif com-video-to-gif-converter (1)](https://github.com/Zzappy24/NBA_V2/assets/102799524/ac24d465-1f86-492c-8b47-3253bdc815f8)


# Use it in local mode

0. Git clone the project

1. Install the required dependencies using the following command:

   ```bash
   pip install -r ./pipeline/requirements.txt ./visualisation/requirements.txt
   ```


2. Open a terminal to execute the script.

3. Run the watcher script:

   ```bash
   cd ./pipeline
   python watcher.py
   ```

4. if you want a better visualisation, open a new terminal and run:

   ```bash
   cd ./visualisation
   python app.py {path to the NBA_V2 folder}
   ```

# If you want to modify the code

1. Do not forget to create a secret variable DOCKER_USERNAME and DOCKER_PASSWORD

2. Change the link of the images in the docker-compose file, and put your images

# Script Overview

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
