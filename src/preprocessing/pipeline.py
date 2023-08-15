from typing import List, Any
from preprocessing.preprocess import *
from config import paths


def create_pipeline(schema: MulticlassClassificationSchema) -> List[Any]:
    """
        Creates pipeline of preprocessing steps

        Args:
            schema (BinaryClassificationSchema): BinaryClassificationSchema object carrying data about the schema
        Returns:
            A list of tuples containing the functions to be executed in the pipeline on a certain column
        """
    pipeline = [
                (indicate_missing_values, None),
                ]
    numeric_features = schema.numeric_features
    cat_features = schema.categorical_features
    for f in numeric_features:
        pipeline.append((impute_numeric, f))
    pipeline.append((normalize, 'schema'))

    for f in cat_features:
        pipeline.append((impute_categorical, f))
    pipeline.append((encode, 'schema'))

    return pipeline


def run_testing_pipeline(data: pd.DataFrame, data_schema: MulticlassClassificationSchema, pipeline: List)\
        -> pd.DataFrame:
    """
    Transforms the data by passing it through every step of the given pipeline.

    Args:
        data (pd.DataFrame): The data to be processed
        data_schema (BinaryClassificationSchema): The schema of the given data.
        pipeline (List): A list of functions to be performed on the data.

    Returns:
        The transformed data
    """

    for stage, column in pipeline:
        if column is None:
            data = stage(data)
        elif column == 'schema':
            if stage.__name__ == 'normalize' and os.path.exists(paths.SCALER_FILE):
                scaler = load(paths.SCALER_FILE)
                data = normalize(data, data_schema, scaler)
            elif stage.__name__ == 'encode':
                data = stage(data, data_schema, encoder='predict')
            else:
                data = stage(data, data_schema)
        else:
            data = stage(data, column)
    return data
