import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, \
     QuantileTransformer, RobustScaler


# ------------------------------------Loading & cleaning------------------------------------- #


def load_data(group):
    """
    Load data from excel and compose a pandas df

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


# ------------------------------------Standardizing & splitting------------------------------------- #


def range_standardizer(x):
    """
    Returns Range standardized datasets set.
    Input: a numpy array, representing entity-to-feature matrix.
    """

    x_rngs = np.ptp(x, axis=0)
    x_means = np.mean(x, axis=0)

    x_r = np.divide(np.subtract(x, x_means), x_rngs)  # range standardization

    return np.nan_to_num(x_r)


def range_standardizer_(x_test, x_train):
    """
    Returns Range standardized datasets set.
    Input: a numpy array, representing entity-to-feature matrix.
    """

    x_rngs = np.ptp(x_train, axis=0)
    x_means = np.mean(x_train, axis=0)

    x_r = np.divide(np.subtract(x_test, x_means), x_rngs)  # range standardization

    return np.nan_to_num(x_r)


def zscore_standardizer(x):
    """
    Returns Z-scored standardized datasets set.
    Input: a numpy array, representing entity-to-feature matrix.
    """

    x_stds = np.std(x, axis=0)
    x_means = np.mean(x, axis=0)

    x_z = np.divide(np.subtract(x, x_means), x_stds)  # z-scoring

    return np.nan_to_num(x_z)


def zscore_standardizer_(x_test, x_train):
    """
    Returns Z-scored standardized datasets set.
    Input: a numpy array, representing entity-to-feature matrix.
    """

    x_stds = np.std(x_train, axis=0)
    x_means = np.mean(x_train, axis=0)

    x_z = np.divide(np.subtract(x_test, x_means), x_stds)  # z-scoring

    return np.nan_to_num(x_z)


def quantile_standardizer(x, out_dist):

    QT = QuantileTransformer(output_distribution=out_dist)
    x_q = QT.fit_transform(x)

    return x_q, QT


def quantile_standardizer_(QT, x):

    x_q = QT.fit_transform(x)

    return x_q


def _minmax_standardizer(x):
    x_mm = MinMaxScaler().fit_transform(x)
    return x_mm


def minmax_standardizer(x):
    x_mm = np.divide(np.subtract(x, x.min(axis=0)),
                     (x.max(axis=0) - x.min(axis=0)))
    return np.nan_to_num(x_mm)


def minmax_standardizer_(x_test, x_train):
    x_mm = np.divide(np.subtract(x_test, x_train.min(axis=0)),
                     (x_train.max(axis=0) - x_train.min(axis=0)))
    return np.nan_to_num(x_mm)


def robust_standardizer(x):
    RS = RobustScaler()
    x_rs = RS.fit_transform(x)
    return x_rs, RS


def robust_standardizer_(RS, x):
    x_rs = RS.fit_transform(x)
    return x_rs


def preprocess_data(x, y, method):

    if method == "rng":
        print("pre-processing:", method)
        x = range_standardizer(x)
        y = range_standardizer(y)
        print("Preprocessed x and y shapes:", x.shape, y.shape)
    elif method == "zsc":
        print("pre-processing:", method)
        x = zscore_standardizer(x)
        y = zscore_standardizer(y)
        print("Preprocessed x and y shapes:", x.shape, y.shape)
    elif method == "mm":  # MinMax
        print("pre-processing:", method)
        x = minmax_standardizer(x)
        y = minmax_standardizer(y)
    elif method == "rs":  # Robust Scaler (subtract median and divide with [q1, q3])
        print("pre-processing:", method)
        x, rs_x = robust_standardizer(x)
        y, rs_y = robust_standardizer(y)
    elif method == "qtn":  # quantile_transformation with Gaussian distribution as output
        x, qt_x = quantile_standardizer(x, out_dist="normal")
        y, qt_y = quantile_standardizer(y, out_dist="normal")
    elif method == "qtu":  # quantile_transformation with Uniform distribution as output
        x, qt_x = quantile_standardizer(x, out_dist="uniform")
        y, qt_y = quantile_standardizer(y, out_dist="uniform")
    elif method is None:
        x_org = x
        y_org = y
        print("No pre-processing")
    else:
        print("Undefined pre-processing")

    return x, y


def _data_index_splitter(x, validation=False):

    if not validation:
        all_idx = np.arange(len(x))
        train_size = int(0.9 * len(all_idx))
        train_idx = np.random.choice(a=all_idx, size=train_size, replace=False, )
        test_idx = list(set(all_idx).difference(train_idx))
        return train_idx, test_idx

    elif validation:
        all_idx = np.arange(len(x))
        train_size = int(0.7 * len(all_idx))
        train_idx = np.random.choice(a=all_idx, size=train_size, replace=False, )
        test_idx = list(set(all_idx).difference(train_idx))
        test_size = int(0.5 * len(test_idx))
        val_idx = np.random.choice(test_idx, size=test_size, replace=False)
        test_idx = list(set(test_idx).difference(val_idx))
        return train_idx, val_idx, test_idx


def data_splitter(x, y, x_org, y_org, target_is_org):

    # train, validation and test split:
    train_idx, val_idx, test_idx = _data_index_splitter(x, validation=True)

    x_train, y_train = x[train_idx, :], y[train_idx, :].ravel()
    x_val, y_val = x[val_idx, :], y[val_idx, :].ravel()
    x_test, y_test = x[test_idx, :], y[test_idx, :].ravel()

    # not preprocessed datasets
    x_train_org, y_train_org = x_org[train_idx, :], y_org[train_idx, :].ravel()
    x_val_org, y_val_org = x_org[val_idx, :], y_org[val_idx, :].ravel()
    x_test_org, y_test_org = x_org[test_idx, :], y_org[test_idx, :].ravel()

    if target_is_org == 1.:
        return x_train, y_train_org, x_val, y_val_org, x_test, y_test_org
    else:
        return x_train, y_train, x_val, y_val, x_test, y_test
