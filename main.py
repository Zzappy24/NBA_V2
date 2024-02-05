import sys

from path.path import CURATED_DATA_DIR_TEMP, TRANFORMED_DATA_DIR_TEMP, MODEL_TEMP_DIR, RAW_DATA_DIR_TEMP, RAW_DATA_DIR

from pathlib import Path
#from path.path import CURATED_DATA_DIR_TEMP, TRANFORMED_DATA_DIR_TEMP, MODEL_TEMP_DIR, RAW_DATA_DIR_TEMP
from DataCleaning.DataCleaningFunctions import write_csv_cleaned, write_csv_cleaned_temp, main_cleaning
from DataTransforming.DataTransformingFunctions import write_csv_tranformed, write_csv_transformed_temp, main_transforming
from DataTraining.DataTrainingFunctions import dump_model, dump_model_temp, main_training
import os
import logging
import time
import shutil
from RaiseError.Error import DuplicateTrainingDataError

log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='Log/pipeline_log.txt', level=logging.INFO, format=log_format)



if __name__ == "__main__":
    succes = True
    step : str
    raws = [f for f in os.listdir(RAW_DATA_DIR) if '.csv' in f.lower()]
    raw_temp = [f for f in os.listdir(RAW_DATA_DIR_TEMP) if '.csv' in f.lower()][0]
    if raw_temp in raws:
        os.remove([f for f in Path(RAW_DATA_DIR_TEMP).iterdir() if f.is_file()][0])
        raise DuplicateTrainingDataError("model already train on this data, be carefull with the choice of the data")

    try :
        logging.info("clean :")
        step = "clean"
        print("clean :")
        df_clean = main_cleaning()
        write_csv_cleaned_temp(df_clean)
        logging.info("cleaning successful")

        logging.info("transform :")
        step = "transform"
        print("transform :")
        df_transformed = main_transforming()
        write_csv_transformed_temp(df_transformed)
        logging.info("transforming successful")
        
        logging.info("train :")
        step = "train"
        print("train :")
        model = main_training()
        dump_model_temp(model)
        logging.info("training successful")

        #logging.info("verification :")
        #logging.info("succes")

    except Exception as e:
        logging.error(f"error in step {step} : {e}")
        succes = False
        print(e)
    
    #time.sleep(10)
    if succes == True:
        file = [f for f in os.listdir(RAW_DATA_DIR_TEMP) if '.csv' in f.lower()][0]
        shutil.move(f"./dataTemp/raw_temp/{file}", f"./data/raw/{file}")
        #Path(f"./dataTemp/raw/{file}").rename(f"./data/raw/{file}")
        write_csv_cleaned(df_clean)
        write_csv_tranformed(df_transformed)
        dump_model(model)


    try :
     os.remove([f for f in Path(RAW_DATA_DIR_TEMP).iterdir() if f.is_file()][0])
    except Exception:
        pass
    try :
        os.remove([f for f in Path(CURATED_DATA_DIR_TEMP).iterdir() if f.is_file()][0])
    except Exception:
        pass
    try :
        os.remove([f for f in Path(TRANFORMED_DATA_DIR_TEMP).iterdir() if f.is_file()][0])
    except Exception:
        pass
    try :
     os.remove([f for f in Path(MODEL_TEMP_DIR).iterdir() if f.is_file()][0])
    except Exception:
        pass
    
    


