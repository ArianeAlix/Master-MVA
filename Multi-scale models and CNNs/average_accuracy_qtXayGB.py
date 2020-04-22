from sklearn.metrics import accuracy_score


import numpy as np

def custom_metric_function(dataframe_1, dataframe_2):
    """
        Example of custom metric function.

    Args
        dataframe_1: Pandas Dataframe
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_1 = pd.read_csv(CSV_1_FILE_PATH, index_col=0, sep=',')

        dataframe_2: Pandas Dataframe
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_2 = pd.read_csv(CSV_2_FILE_PATH, index_col=0, sep=',')

    Returns
        score: Float
            The metric evaluated with the two dataframes. This must not be NaN.
    """

    score = np.sum(dataframe_1["sexe"]==dataframe_2["sexe"])
    score += np.sum(dataframe_1["date_accident"]==dataframe_2["date_accident"])
    score += np.sum(dataframe_1["date_consolidation"]==dataframe_2["date_consolidation"])
    score /= (3*dataframe_1.shape[0])

    return score


if __name__ == '__main__':
    import pandas as pd
    CSV_FILE_1 = 'Y_test.csv'
    CSV_FILE_2 = 'prediction.csv'
    df_1 = pd.read_csv(CSV_FILE_1, index_col=0, sep=',')
    df_2 = pd.read_csv(CSV_FILE_2, index_col=0, sep=',')
    print(custom_metric_function(df_1, df_2))
