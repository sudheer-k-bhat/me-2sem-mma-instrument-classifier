from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.naive_bayes import GaussianNB
import librosa
import numpy as np
import pandas as pd

class InstrumentClassifier:
    sr = 44100
    random_state = 42
    def __init__(self, n_samples, feature = 'mfcc', model='gnb', pca_components = 20):
        self.n_samples = n_samples
        self.feature = feature
        self.model = model

        if 'gnb' == model:
            self.pipeline = Pipeline([
                ('Standard Scaler', StandardScaler()),
                ('PCA', PCA(n_components = pca_components)),
                ('Gaussian Naive Bayes', GaussianNB())
            ])
        
        if 'svc' == model:
            self.pipeline = Pipeline([
                ('Standard Scaler', StandardScaler()),
                ('PCA', PCA(n_components = pca_components)),
                ('SVM', svm.SVC(gamma = 'auto'))
            ])

    def _extract(self, row):
        signal, sr = librosa.load(f'wavfiles/{row[0]}', InstrumentClassifier.sr)
        if 'mfcc' in self.feature:
            X = row['mfcc']
        if 'melspectrogram' in self.feature:
            X = row['melspectrogram']
        X_y = np.append(X, row[1])
        return X_y

    def _extract_X_y(self, df):
        samples = df.sample(n=self.n_samples, replace=True, random_state=InstrumentClassifier.random_state)
        data = samples.apply(self._extract, axis=1, result_type='expand')
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]
        return X, y

    def extract_features(self, df):
        self.X, self.y = self._extract_X_y(df)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=InstrumentClassifier.random_state)

    def fit(self):
        self.pipeline.fit(self.x_train, self.y_train)
    
    def perf(self):
        return self.pipeline.score(self.x_test, self.y_test)
    
    def predict(self):
        self.y_pred = self.pipeline.predict(self.x_test)