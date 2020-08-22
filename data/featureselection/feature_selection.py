from skrebate import ReliefF
from sklearn.feature_selection import mutual_info_classif
import pymrmr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
from data.featureselection.gmm import processing_gmm
from operator import itemgetter


class FeatureSelection:

    # On init, set method for use as pca, lda, relieff, ig, mrmr, or rf
    def __init__(self, method):
        self.method = method

    # Perform PCA given any pandas dataframe
    @staticmethod
    def processing_pca(df, n_components):

        df = df.dropna()
        x = df.drop(['DX'], axis=1)
        y = df['DX']
        sc = StandardScaler()
        x = sc.fit_transform(x)
        pca = PCA(n_components=n_components)
        x = pca.fit_transform(x)
        explained_variance = pca.explained_variance_ratio_

        tmp = pd.DataFrame(x)
        return pd.concat([tmp, y])

    # Perform LDA given any pandas dataframe
    @staticmethod
    def processing_lda(df, n_components):

        df = df.dropna()
        x = df.drop(['DX'], axis=1)
        y = df['DX']
        sc = StandardScaler()
        x = sc.fit_transform(x)
        lda = LDA(n_components=n_components)
        x = lda.fit_transform(x, y)
        # x_test = lda.transform(x_test) #TODO build a transform method
        explained_variance = lda.explained_variance_ratio_

        tmp = pd.DataFrame(x)
        return pd.concat([tmp, y])

    @staticmethod
    def processing_relieff(df, n_components):

        features_selected = ReliefF()
        x, y = df.drop('DX', axis=1).values, df['DX'].values

        features_selected.fit(x, y)

        relief_dict = dict(zip(df.drop('DX', axis=1).columns, features_selected.feature_importances_))
        top_features = dict(sorted(relief_dict.items(), key=itemgetter(1), reverse=True)[:n_components]).keys()

        top_features = list(top_features)
        if 'DX' not in top_features:
            top_features.append('DX')

        return df[top_features], top_features

    @staticmethod
    def processing_information_gain(df, n_components):

        x, y = df.drop('DX', axis=1).values, df['DX'].values

        ig_dict = dict(zip(df.drop('DX', axis=1),
                           mutual_info_classif(x, y, discrete_features=True)))

        top_features = dict(sorted(ig_dict.items(), key = itemgetter(1), reverse = True)[:n_components]).keys()

        top_features = list(top_features)
        if 'DX' not in top_features:
            top_features.append('DX')

        return df[top_features], top_features

    # Perform MRMR with MIQ given any pandas dataframe
    @staticmethod
    def processing_mrmr(df, n_components, mrmr_method='MIQ'):

        top_features = pymrmr.mRMR(df, mrmr_method, n_components)

        if 'DX' in top_features:
            n_components = n_components + 1
            print('Issue with MRMR - need next feature')
            top_features = pymrmr.mRMR(df, mrmr_method, n_components)

        if 'DX' not in top_features:
            top_features.append('DX')

        return df[top_features], top_features

    # Perform Random Forest Feature Importance
    @staticmethod
    def processing_random_forest(df, n_components, rs, num_trees=100):

        x = df.drop(['DX'], axis=1)
        y = df['DX']

        # Build a forest and compute the impurity-based feature importances
        forest = ExtraTreesClassifier(n_estimators=num_trees, random_state=rs)

        forest.fit(x, y)

        feature_importances = pd.DataFrame(forest.feature_importances_,
                                           index=x.columns,
                                           columns=['importance']).sort_values('importance', ascending=False)

        print(feature_importances)

        top_features = feature_importances.head(n_components)
        top_features = list(top_features.index.values)

        if 'DX' not in top_features:
            top_features.append('DX')

        return df[top_features], top_features

    @staticmethod
    def processing_univariate(df, n_components):

        x = df.drop(['DX'], axis=1)
        y = df['DX']

        kbest = SelectKBest(score_func=f_classif, k=n_components)
        kbest.fit_transform(x,y)
        feature_array = list(kbest.get_support())

        top_features = [col for col, keep in zip(x.columns, feature_array) if keep]
        if 'DX' not in top_features:
            top_features.append('DX')

        return df[top_features], top_features

    def choose_features(self, data, rs, n_components):

        # Use GMM to find optimum number of components
        #n_components = processing_gmm(data)
        n_components = int(n_components)

        if n_components == 0:
            n_components = 1

        if self.method == 'relieff':
            return self.processing_relieff(data, n_components)
        elif self.method == 'ig':
            return self.processing_information_gain(data, n_components)
        elif self.method == 'mrmr':
            return self.processing_mrmr(data, n_components)
        elif self.method == 'rf':
            # Trying 38 from previous literature
            return self.processing_random_forest(data, n_components, rs, 38)
        elif self.method == 'univariate':
            return self.processing_univariate(data, n_components)
        return data, ['All']
