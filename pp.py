import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn import model_selection as ms
from sklearn import svm

def DelMissingValues(df, axis=0):
    rows = df.values.tolist()
    clean_rows = []
    index = []

    for row in rows:
        flag = True
        for i in range(len(row)):
            if str(row[i]) == 'nan':
                flag = False
                index.append(i)
                break
        if flag:
            clean_rows.append(row)

    if axis == 0:
        return pd.DataFrame.from_records(clean_rows)
    else:
        return  pd.DataFrame.from_records(rows).drop(set(index),axis=1)


def ReplaceWithMode(df):
    columns = []
    for i in range(len(df.columns)):
        column = np.array(df[i])
        uniqe_per_column = list(set(column))
        count_of_elem = [0]*len(uniqe_per_column)

        for i in range(len(uniqe_per_column)):
            if str(uniqe_per_column[i]) == 'nan':
                continue
            else:
                for j in range(len(column)):
                    if uniqe_per_column[i] == column[j]:
                        count_of_elem[i] += 1

        maximum = count_of_elem[0]
        index = 0

        for i in range(1,len(count_of_elem)):
            if maximum < count_of_elem[i]:
                maximum = count_of_elem[i]
                index = i

        mode = uniqe_per_column[index]

        for i in range(len(column)):
            if str(column[i]) == 'nan':
                column[i] = mode
        columns.append(column)

    return pd.DataFrame.from_records(np.transpose(columns))

def ReplaceWithMedian(df):
    columns = []
    for i in range(len(df.columns)):
        column = np.array(df[i])
        indices = np.where(np.logical_not(np.isnan(column)))[0]
        median = np.median(column[indices])
        for i in range(len(column)):
            if str(column[i]) == 'nan':
                column[i] = median
        columns.append(column)
    return pd.DataFrame.from_records(np.transpose(columns))


def ReplaceWithAverage(df):
    columns = []
    for i in range(len(df.columns)):
        column = np.array(df[i])
        indices = np.where(np.logical_not(np.isnan(column)))[0]
        avg = np.average(column[indices])
        for i in range(len(column)):
            if str(column[i]) == 'nan':
                column[i] = avg
        columns.append(column)
    return pd.DataFrame.from_records(np.transpose(columns))

def Standartization(df):
    columns = []
    for i in range(len(df.columns)-1):
        column = df[i]
        column_std = np.nanstd(column)
        column_mean = np.nanmean(column)

        for j in range(len(column)):
            if str(column[j]) != 'nan':
                column[j] = (column[j] - column_mean)/column_std
        columns.append(column)
    columns.append(df[5])
    return pd.DataFrame.from_records(np.transpose(columns))


def Scale(df):
    columns = []
    for i in range(len(df.columns)):
        column = df[i]
        column_min = np.min(column)
        column_max = np.max(column)
        for j in range(len(column)):
            if str(column[j]) != 'nan':
                column[j] = (column[j] - column_min) / (column_max - column_min)
        columns.append(column)

    return pd.DataFrame.from_records(np.transpose(columns))