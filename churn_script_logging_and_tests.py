"""
Module that runs tests for churn.librayr.py and save results on log file

Author: AlexGM
Date: Dec 2021
"""

import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with
    the other test functions
    '''
    try:
        data_frame = import_data("data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError:
        logging.error("Testing import_eda: The file wasn't found")

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")

    return data_frame


def test_eda(perform_eda, data_frame):
    '''
    test perform eda function
    '''
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    perform_eda(data_frame)
    path = "images/eda"

    try:
        assert data_frame["Churn"].isnull().sum() == 0
        logging.info("Testing test eda: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing test eda: Churn column has one or more NAN values")

    try:
        im_path = os.listdir(path)
        assert len(im_path) == 5
        logging.info("Testing peform_eda: SUCCESS")
    except AssertionError:
        logging.error("Testing peform_eda: One or more EDA plots missing")


def test_encoder_helper(encoder_helper, data_frame):
    '''
    test encoder helper
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    data_frame = encoder_helper(data_frame, cat_columns, 'Churn')

    try:
        for col in cat_columns:
            assert f"{col}_Churn" in data_frame.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing encoder_helper: Missing one or more transformed cat columns")

    return data_frame


def test_perform_feature_engineering(perform_feature_engineering, data_frame):
    '''
    test perform_feature_engineering
    '''
    X_train, X_test, y_train, y_test = perform_feature_engineering(data_frame)
    try:
        assert len(X_train) > 0
        assert len(y_train) > 0
        assert len(X_test) > 0
        assert len(y_test) > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing perform_feature_engineering: One or more resulting objects are empty")

    return X_train, X_test, y_train, y_test


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''
    train_models(X_train, X_test, y_train, y_test)
    im_path = "images/results/"
    try:
        dir_val = os.listdir(im_path)
    except FileNotFoundError:
        logging.error(
            "Testing train_models: image results directory not found")

    try:
        dir_val = os.listdir(im_path)
        assert len(dir_val) == 4
    except AssertionError:
        logging.error("Testing train_models: model result plots not found")

    model_path = "models/"
    try:
        dir_val = os.listdir(model_path)
    except FileNotFoundError:
        logging.error("Testing train_models: model files directory not found")
    try:
        dir_val = os.listdir(model_path)
        assert len(dir_val) == 2
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError:
        logging.error("Testing train_models: model files not found")


if __name__ == "__main__":
    DATA_FRAME = test_import(cls.import_data)
    test_eda(cls.perform_eda, DATA_FRAME)
    DATA_FRAME = test_encoder_helper(cls.encoder_helper, DATA_FRAME)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cls.perform_feature_engineering,
        DATA_FRAME
    )
    test_train_models(cls.train_models, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
