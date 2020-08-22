from sklearn.utils import compute_class_weight

from data.load_transform.load import Load
from data.load_transform.transform import Transform
from data.featureselection.feature_selection import FeatureSelection
from data.load_transform.data_splitting import split
from data.models.model import Model
from data.plots.visualizations import Visualizations
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class Pipeline(object):

    pipeline_config = None

    def __init__(self, pipeline_config, random_state):
        self.pipeline_config = pipeline_config
        self.random_state = random_state

    @staticmethod
    def build_data():
        # Load Data
        data_loader = Load()
        clinical_data, mri_data, pet_data, sociodemographic_data, fluid_data = data_loader.load()
        # Clean Data
        data_transformer = Transform(clinical_data, mri_data, pet_data, sociodemographic_data, fluid_data)
        clinical_data, mri_data, pet_data, sociodemographic_data, fluid_data = data_transformer.clean()

        return {'mri': mri_data.toPandas(), 'pet': pet_data.toPandas(), 'clinical': clinical_data.toPandas(),
                'sociodemographic': sociodemographic_data.toPandas(), 'fluid': fluid_data.toPandas()}

    def feature_selection(self, data_list):
        data_list_final = []
        feature_list_final = []
        for data_set, feature_selection_method, n_components in zip(data_list, self.pipeline_config['feature_selection'], self.pipeline_config['n_components']):
            # Feature Selection
            fs = FeatureSelection(feature_selection_method)
            data, features = fs.choose_features(data_set[1], self.random_state, n_components)
            data_list_final.append((data_set[0], data))
            feature_list_final.append((data_set[0], features))
        return data_list_final, feature_list_final

    @staticmethod
    def build_final_dataset(data_list):

        full_data = pd.concat([data_list['pet'], data_list['mri'].drop('DX', axis=1),
                               data_list['clinical'].drop('DX', axis=1), data_list['fluid'].drop('DX', axis=1),
                               data_list['sociodemographic'].drop('DX', axis=1)], axis=1)

        image_data = pd.concat([data_list['pet'], data_list['mri'].drop('DX', axis=1)], axis=1)

        non_image_data = pd.concat([data_list['clinical'], data_list['fluid'].drop('DX', axis=1),
                                    data_list['sociodemographic'].drop('DX', axis=1)], axis=1)

        data_list['full_data'] = full_data
        data_list['imaging'] = image_data
        data_list['non_imaging'] = non_image_data

        return data_list

    def build_data_list(self, data):
        return [(data_set, data[data_set]) for data_set in self.pipeline_config['data_list']]

    def run_model(self, data_list, rs):
        # Build classifier and dataset lists
        classifier_list = self.pipeline_config['classifiers']

        model = Model(self.pipeline_config)
        model.reset_random_seeds(rs)

        # Get initial k_fold indicies for future splits to be uniform
        folds = int(self.pipeline_config['folds'])

        data_list_split = []
        for data_set in data_list:
            # Split Data, store test data for later
            data_list_split.append((data_set[0], split(data_set[1], self.pipeline_config['data_split'],
                                                       self.random_state, folds)))

        metrics = {'accuracy': 0, 'auc': 0, 'sensitivity': 0, 'specificity': 0}
        for fold in range(1, folds+1):
            print('Running Fold {}'.format(fold))
            x_train_list, y_train_list, x_test_list, y_test_list, model_list = [], [], [], [], []
            for data, classifier in zip(data_list_split, classifier_list):
                x_train, y_train, x_test, y_test = data[1][str(fold)]
                dim = len(x_train.columns)

                sc = StandardScaler()
                x_train = sc.fit_transform(x_train)
                x_test = sc.fit_transform(x_test)

                x_train_list.append(x_train)
                y_train_list.append(y_train)
                x_test_list.append(x_test)
                y_test_list.append(y_test)

                class_weight = compute_class_weight('balanced', np.arange(5), y_train)
                class_weight = dict(enumerate(class_weight))
                model_list.append(model.build(classifier, dim, class_weight)[0])

            model_list_trained, label_encoder = model.train_multiple_classifiers(x_train_list, y_train_list[0],
                                                                                 model_list, class_weight)

            metrics_ = model.predict_from_multiple_classifiers(x_test_list, y_test_list, model_list_trained,
                                                               label_encoder)

            metrics['accuracy_fold_{}'.format(fold)] = metrics_['accuracy']
            metrics['auc_fold_{}'.format(fold)] = metrics_['auc']
            metrics['sensitivity_fold_{}'.format(fold)] = metrics_['sensitivity']
            metrics['specificity_fold_{}'.format(fold)] = metrics_['specificity']

            metrics['accuracy'] = metrics['accuracy'] + metrics_['accuracy']
            metrics['auc'] = metrics['auc'] + metrics_['auc']
            metrics['sensitivity'] = metrics['sensitivity'] + metrics_['sensitivity']
            metrics['specificity'] = metrics['specificity'] + metrics_['specificity']

        # Get average of metrics from all runs
        metrics['accuracy'] = metrics['accuracy'] / folds
        metrics['auc'] = metrics['auc'] / folds
        metrics['sensitivity'] = metrics['sensitivity'] / folds
        metrics['specificity'] = metrics['specificity'] / folds

        # Return metrics
        return metrics

    def build_visualizations(self, results, experiment):

        vis = Visualizations(results, experiment)
        vis.build_whisker_plot()
        vis.build_scatter_plot()
