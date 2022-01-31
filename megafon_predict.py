import pandas as pd
import numpy as np
import dask.dataframe as dd
import sys
import joblib
import os

from datetime import datetime
import sklearn
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one
from scipy import sparse


class PandasFeatureUnion(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans,
                X=X,
                y=y,
                weight=weight,
                **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[[self.key]].rename(columns={self.key: 'cat/' + self.key})


class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]].rename(columns={self.key: 'nmb/' + self.key})


class PolynomSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """

    def __init__(self, key, degree=2):
        self.key = key
        self.degree = degree
        self.column = 'pol' + str(degree) + '/' + self.key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame({self.column: X[self.key] ** self.degree})


class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=StandardScaler):
        self.scaler = scaler()
        self.columns = []

    def fit(self, X, y=None):
        self.columns = X.columns.tolist()
        self.scaler.fit(X)
        return self

    def transform(self, X, y=None):
        index = X.index
        return pd.DataFrame(self.scaler.transform(X), columns=self.columns, index=index)


class OHEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, values=[]):
        self.key = []
        self.columns = []
        self.values = []
        if len(values) > 0:
            self.values = [str(val) for val in values]

    def fit(self, X, y=None):
        self.key = X.columns.tolist()
        if len(self.values) == 0:
            self.columns = [col for col in pd.get_dummies(X, prefix=self.key, prefix_sep='/', columns=self.key).columns]
        else:
            self.columns = [col for col in pd.get_dummies(X, prefix=self.key, prefix_sep='/', columns=self.key).columns
                            if col.split('/')[2] in self.values]
        return self

    def transform(self, X):
        X_ = pd.get_dummies(X, prefix=self.key, prefix_sep='/', columns=self.key)
        test_columns = [col for col in X_.columns]
        for col_ in self.columns:
            if col_ not in test_columns:
                X_[col_] = 0
        return X_[self.columns]


class MonthSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """

    def __init__(self, key):
        self.key = key
        self.column = 'month/' + self.key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X[self.key].apply(lambda x: datetime.fromtimestamp(x).month)
        return pd.DataFrame({self.column: X_})


def data_features_merge(input_file, tmp_file):
    data = pd.read_csv(input_file)
    data = data.rename(columns={'Unnamed: 0': 'offer_id'})

    data_features = dd.read_csv('features.csv', sep='\t')
    data_features = data_features.rename(columns={'buy_time': 'feat_time'})
    data_features = data_features.drop(['Unnamed: 0'], axis=1)

    df_merge = data_features.merge(data, how='right', on=['id'])
    df_merge.to_csv(tmp_file, single_file=True)

    return


def filter_features(input_file, tmp_file):
    data = pd.read_csv(input_file)
    data = data.drop(['Unnamed: 0'], axis=1)

    data['lag'] = data['buy_time'] - data['feat_time']
    # Для записей в которых дата продажи раньше даты признаков устанавливаем максимальный лаг+отклонение
    max_lag = data['lag'].max()
    msk = data['lag'] < 0
    data.loc[msk, 'lag'] = max_lag - data['lag']

    grp = data.groupby('offer_id').agg({'lag': min}).reset_index()
    data = grp.merge(data, how='left', on=['offer_id', 'lag'])

    data.to_csv(tmp_file)
    return


def model_predict(model_file, input_file, output_file):
    data = pd.read_csv(input_file)
    data = data.set_index('offer_id')
    estimator = joblib.load(model_file)

    columns = []
    for col in data.columns:
        if col.isnumeric():
            columns += ['feat-'+col]
            data[col] = data[col].astype(np.float32)
        else:
            columns += [col]

    data.columns = columns

    y_predict = estimator.predict_proba(data)[:, 1]

    data['target'] = y_predict
    data.index.names = ['']
    data.loc[:, ['buy_time', 'id', 'vas_id', 'target']].to_csv(output_file)
    return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("python megafon_predict data_file_name [model_file_name]")
        exit(1)
    output_file = os.path.join(os.path.dirname(sys.argv[1]), 'answers_test.csv')


    data_features_merge(sys.argv[1], 'merge_tmp.csv')
    filter_features('merge_tmp.csv', 'data_tmp.csv')
    model_file_name = 'log_reg_model3.pkl' if len(sys.argv) < 3 else sys.argv[2]
    model_predict(model_file_name, 'data_tmp.csv', output_file)
    os.remove('merge_tmp.csv')
    os.remove('data_tmp.csv')



