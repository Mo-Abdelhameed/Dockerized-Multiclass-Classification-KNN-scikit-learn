import os
from Classifier import Classifier
from utils import read_csv_in_directory
from config import paths
from logger import get_logger, log_error
from schema.data_schema import load_json_data_schema, save_schema
from preprocessing.pipeline import create_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from utils import set_seeds
from joblib import dump


logger = get_logger(task_name="train")


def run_training(
        input_schema_dir: str = paths.INPUT_SCHEMA_DIR,
        saved_schema_dir_path: str = paths.SAVED_SCHEMA_DIR_PATH,
        train_dir: str = paths.TRAIN_DIR,
        predictor_dir_path: str = paths.PREDICTOR_DIR_PATH,
       ) -> None:
    """
    Run the training process and saves model artifacts

    Args:
        input_schema_dir (str, optional): The directory path of the input schema.
        saved_schema_dir_path (str, optional): The path where to save the schema.
        train_dir (str, optional): The directory path of the train data.
        predictor_dir_path (str, optional): Dir path where to save the
            predictor model.
    Returns:
        None
    """
    try:
        logger.info("Starting training...")
        set_seeds(seed_value=123)

        logger.info("Loading and saving schema...")
        data_schema = load_json_data_schema(input_schema_dir)
        save_schema(schema=data_schema, save_dir_path=saved_schema_dir_path)

        logger.info("Loading training data...")
        train_data = read_csv_in_directory(train_dir)
        features = data_schema.features
        target = data_schema.target
        x_train = train_data[features]
        y_train = train_data[target]
        pipeline = create_pipeline(data_schema)
        for stage, column in pipeline:
            if column is None:
                x_train = stage(x_train)
            elif column == 'schema':
                x_train = stage(x_train, data_schema)
            else:
                if stage.__name__ == 'remove_outliers_zscore':
                    x_train, y_train = stage(x_train, column, target=y_train)
                else:
                    x_train = stage(x_train, column)
        dump(x_train.columns, paths.TRAIN_COLUMNS_FILE)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)
        best_score = 0
        best_n = 1
        for i in range(1, 30, 2):
            model = Classifier(n_neighbors=i)
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            try:
                score = f1_score(y_test, predictions, average='weighted')
            except ValueError:
                score = f1_score(y_test, predictions, pos_label=data_schema.target_classes[1])
            if score > best_score:
                best_score = score
                best_n = i
        model = Classifier(n_neighbors=best_n)
        model.fit(x_train, y_train)
        if not os.path.exists(predictor_dir_path):
            os.makedirs(predictor_dir_path)
        model.save(predictor_dir_path)
        logger.info(f'Model saved: KNeighbourClassifier n={best_n}.')

    except Exception as exc:
        err_msg = "Error occurred during training."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.TRAIN_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    run_training()
