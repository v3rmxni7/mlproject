import os
import sys
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.exception import CustomException
from src.logger import logging


from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


# Configuration for data ingestion file paths
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")

# Data Ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  # ✅ FIXED

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")

        try:
            # Load raw data
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'notebook', 'data', 'stud.csv'))
            logging.info("Dataset loaded successfully")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split into train/test
            logging.info("Starting train/test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

# Run when script is executed directly
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data= obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,preprocessor_path =data_transformation.initiate_data_transformation(train_data,test_data)


    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr,preprocessor_path))