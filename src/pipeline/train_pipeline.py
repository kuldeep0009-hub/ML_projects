from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    # Data Ingestion
    ingestion = DataIngestion()
    train_data_path, test_data_path = ingestion.initiate_data_ingestion()

    # Data Transformation
    transformer = DataTransformation()
    train_arr, test_arr, _ = transformer.initiate_data_transformation(train_data_path, test_data_path)

    # Model Training
    trainer = ModelTrainer()
    print(trainer.initiate_model_trainer(train_arr, test_arr))
