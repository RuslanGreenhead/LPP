import numpy as np
import pandas as pd
#import utilities as util
from sklearn.preprocessing import MinMaxScaler, \
    StandardScaler, QuantileTransformer, RobustScaler


# Load data from excel and compose a pandas df
def load_data(group):
    """
    :param:
        group: string, task-relation : classification / regression
    :return:
        data_org := pandas df, not preprocessed
        x := numpy, independent variables (features)
        y := numpy, dependent variables (target values)
        indicators := list, a list of subject id, sentence etc. depending on the dataset structure

    """

    ia_df = pd.read_excel("../datasets/IA_report.xlsx")
    demo_df = pd.read_excel("../datasets/demo.xlsx")

    if "classification" in group:
        target = "English_Level"
    elif "regression" in group:
        target = 'L1_spelling_skill'
    else:
        raise RuntimeError("Unknown group of experiment")

    demo_df.set_index('SubjectID', inplace=True)
    ia_df[['Age', 'Sex(0-f,1-m)', 'IQ', target]] = 0, 0, 0, 0

    df = ia_df.apply(_col_filler, axis=1, raw=False, df=demo_df, col='Age')
    df = df.apply(_col_filler, axis=1, raw=False, df=demo_df, col='Sex(0-f,1-m)')
    df = df.apply(_col_filler, axis=1, raw=False, df=demo_df, col='IQ')
    df = df.apply(_col_filler, axis=1, raw=False, df=demo_df, col=target)
    df = remove_missing_data(df)

    cols = list(df.columns)
    cols = [cols[0]] + [cols[-1]] + cols[-4:-1] + cols[1:-4]
    df = df[cols]
    df = df.apply(check_col, axis=0, group=group)
    df = df.dropna(axis=0)

    b_features = ['SKIP', 'Sex(0-f,1-m)', 'REGRESSION_IN', 'REGRESSION_OUT', 'REGRESSION_OUT_FULL']
    q_features = ['Age', 'IQ', 'FIXATION_COUNT', 'TOTAL_READING_TIME',
                  'FIRST_FIXATION_DURATION', 'FIRST_FIXATION_X', 'FIRST_FIXATION_Y',
                  'FIRST_RUN_TOTAL_READING_TIME', 'FIRST_SACCADE_AMPLITUDE',
                  'REGRESSION_PATH_DURATION']
    features = b_features + q_features
    indexes = ['SubjectID', 'Test_ID', 'Word_Number']

    x = df[features].values
    y = df[target].values

    return df, x, y, features, target, indexes


def _col_filler(x, df, col):
    """
    Function to fill cell in a row with corresponding value from demo_df (SubjectID as a link).
    Takes row + name of column to fill, returns transformed row

    :param x: pd.Series, object as a row
           df: pd.DataFrame to take data from
           col: feature we want to fill with value
    :return: filled row
    """
    res_series = x
    res_series[col] = df.loc[x.SubjectID, col]
    return res_series


def check_col(col, group):
    if "classification" in group:
        if col.name not in ['SubjectID', 'English_Level']:
            return pd.to_numeric(col, errors='coerce')
    elif "regression" in group:
        if col.name not in ['SubjectID',]:
            return pd.to_numeric(col, errors='coerce')
    return col


def remove_missing_data(df):
    for col in df.columns:
        try:
            df[col].replace({".": np.nan}, inplace=True)
        except Exception as e:
            print(e, "\n No missing values in", col)

    return df.dropna()


