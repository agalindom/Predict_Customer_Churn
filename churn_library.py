# library doc string
"""
Module that holds processing steps

Author: AlexGM
Date: Dec 2021
"""

# import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    """
    data_frame = pd.read_csv("%s" % pth)
    return data_frame


def perform_eda(data_frame):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    # Start EDA plots
    plt.figure(figsize=(20, 10))
    data_frame['Churn'].hist()
    plt.title("Churn Histogram")
    plt.savefig("images/EDA/Churn_Hist.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    data_frame['Customer_Age'].hist()
    plt.title("Customer Age Histogram")
    plt.savefig("images/EDA/Customer_Age_Hist.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title("Marital Status Value Counts")
    plt.savefig("images/EDA/Marital_Status_Counts.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.displot(data_frame['Total_Trans_Ct'])
    plt.title("Distribution of Total Trans")
    plt.savefig("images/EDA/Total_Trans_Dist.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title("Feature Heatmap")
    plt.savefig("images/EDA/Feature_Heatmap.png")
    plt.close()


def encoder_helper(data_frame, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of categorical column features
            response: string of response name [optional]

    output:
            data_frame: pandas dataframe with new columns for
    '''
    for cat_col in category_lst:
        cat_col_list = []
        cat_col_groups = data_frame.groupby(cat_col).mean()[response]

        for val in data_frame[cat_col]:
            cat_col_list.append(cat_col_groups.loc[val])

        new_col_name = "%s_%s" % (cat_col, response)
        data_frame[new_col_name] = cat_col_list

    return data_frame


def perform_feature_engineering(data_frame, response=None):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    if response is None:
        response = "Churn"
    x_data = encoder_helper(data_frame, cat_columns, response)
    x_data = x_data[keep_cols]
    x_data = normalize(x_data)
    x_data = pd.DataFrame(x_data, columns=keep_cols)
    y_data = data_frame["Churn"].values
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        x_data, y_data,
        test_size=0.3,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Random Forest
    plt.figure(figsize=(7, 7))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.title("Random Forest Train", loc="left")
    plt.text(
        0.01, 0.3,
        str(classification_report(y_test, y_test_preds_rf, zero_division=0)),
        {'fontsize': 10},
        fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(
        0.01, 0.7,
        str('Random Forest Test'),
        {'fontsize': 10},
        fontproperties='monospace')
    plt.text(
        0.01, 0.9,
        str(classification_report(y_train, y_train_preds_rf, zero_division=0)),
        {'fontsize': 10},
        fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("images/results/Classification_Report_RF.png")
    plt.close()

    # Logistic regression
    plt.figure(figsize=(7, 7))
    plt.title("Logistic Regression Train", loc="left")
    plt.text(
        0.01, 0.3,
        str(classification_report(y_train, y_train_preds_lr, zero_division=0)),
        {'fontsize': 10},
        fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(
        0.01, 0.7,
        str('Logistic Regression Test'),
        {'fontsize': 10},
        fontproperties='monospace')
    plt.text(
        0.01, 0.9,
        str(classification_report(y_test, y_test_preds_lr, zero_division=0)),
        {'fontsize': 10},
        fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("images/results/Classification_Report_LR.png")
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 9))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    plt.close


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # Create and save ROC curves
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/results/roc_curve_result.png')
    plt.close()

    feature_importance_plot(
        cv_rfc,
        X_train,
        "images/results/Feature_Importance_Plot.png"
    )

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == "__main__":
    DATA_FRAME = import_data("data/bank_data.csv")
    perform_eda(DATA_FRAME)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(DATA_FRAME)
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
