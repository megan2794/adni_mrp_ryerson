from tensorflow.keras.metrics import AUC, CategoricalAccuracy
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

import os
import tensorflow as tf
import random


class Model(object):

    pipeline_config = None

    def __init__(self, pipeline_config):
        self.pipeline_config = pipeline_config

    @staticmethod
    def reset_random_seeds(rs):
        os.environ['PYTHONHASHSEED'] = str(rs)
        tf.random.set_seed(rs)
        np.random.seed(rs)
        random.seed(rs)

    @staticmethod
    def get_metrics(y, y_pred):

        cnf_matrix = confusion_matrix(y, y_pred)

        false_positive = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix).astype(float)
        false_negative = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix).astype(float)
        true_positive = np.diag(cnf_matrix).astype(float)
        true_negative = (cnf_matrix.sum() - (false_positive + false_negative + true_positive)).astype(float)


        y = to_categorical(y, num_classes=3)
        y_pred = to_categorical(y_pred, num_classes=3)

        auc = AUC()
        _ = auc.update_state(y, y_pred)
        acc = CategoricalAccuracy()
        _ = acc.update_state(y, y_pred)

        return {'accuracy': acc.result().numpy(), 'auc': auc.result().numpy(),
                'sensitivity': true_positive / (true_positive + false_negative),
                'specificity': true_negative / (true_negative + false_positive)}

    def predict_from_multiple_classifiers(self, x_list, y_test, model_list, label_encoder):

        predictions = np.asarray([model.predict(X) for model, X in zip(model_list, x_list)])
        #prediction_avg = np.average(predictions, axis=0)
        prediction_avg = np.mean(predictions, axis=0)
        pred = np.argmax(prediction_avg, axis=1)

        # Convert integer predictions to original labels:
        y_pred = label_encoder.inverse_transform(pred)

        return self.get_metrics(np.asarray(y_test[0]), y_pred)

    @staticmethod
    def train_multiple_classifiers(x_list, y, model_list, class_weights):
        le_ = LabelEncoder()
        le_.fit(y)

        estimators_ = [model.train(X, y, class_weights) for model, X in zip(model_list, x_list)]
        return estimators_, le_

    def build(self, classifier, dim, class_weights):

        model_list = []
        if 'dl' in classifier:
            dl_model = DeepLearningModel(self.pipeline_config)
            model_list.append(dl_model.build(classifier, dim, class_weights))
        else:
            traditional_model = TraditionalModel(self.pipeline_config)
            model_list.append(traditional_model.build(classifier, dim, class_weights))

        return model_list


class DeepLearningModel(Model):

    def __init__(self, pipeline_config):
        self.classifier_name = None
        self.model = None
        super(DeepLearningModel, self).__init__(pipeline_config)

    def build(self, classifier, dim, class_weights):


        layers = self.pipeline_config['layers']
        dropouts = self.pipeline_config['dropouts']

        model = Sequential()
        model.add(Dense(int(layers[0][0]), input_shape=(dim,), activation=str(layers[0][1])))

        for layer, dropout in zip(layers, dropouts):
            if float(dropout) > 0:
                model.add(Dropout(float(dropout)))
            model.add(Dense(int(layer[0]), activation=str(layer[1])))

        mets = [str(self.pipeline_config['accuracy'])]
        #[str(self.pipeline_config['accuracy'])]
        #sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=str(self.pipeline_config['loss']), optimizer=str(self.pipeline_config['opt']),
                      metrics=mets)

        self.classifier_name = classifier
        self.model = model
        return self

    def train(self, x_train, y_train, class_weight):
        #class_weight = compute_class_weight('balanced', np.arange(5), y_train)
        #class_weight = dict(enumerate(class_weight))
        y_train = to_categorical(y_train, num_classes=3)
        #x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

        #print(class_weight)
        self.model.fit(x_train, y_train, epochs=int(self.pipeline_config['epochs']),
                       batch_size=int(self.pipeline_config['batch_size']))  # , class_weight=class_weight)
        return self

    def predict(self, x_test):
        #x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        y_pred = self.model.predict(x_test)
        y_pred =(y_pred == y_pred.max(axis=1, keepdims=1)).astype(float)
        return y_pred


class TraditionalModel(Model):

    def __init__(self, pipeline_config):
        self.model = None
        self.classifier_name = None
        super(TraditionalModel, self).__init__(pipeline_config)

    def build(self, classifier, dim, class_weights):

        if classifier.split('_')[0] == 'knn':
            self.model = KNeighborsClassifier(2)
        elif classifier.split('_')[0] == 'nb':
            self.model = GaussianNB()
        elif classifier.split('_')[0] == 'svc':
            self.model = SVC(kernel="rbf", cache_size=1000000,
                             C=0.0016, probability=True) #,  # class_weight=class_weights)
        elif classifier.split('_')[0] == 'xgb':
            self.model = XGBClassifier()

        self.classifier_name = classifier
        return self

    def train(self, x_train, y_train, class_weights):
        self.model = self.model.fit(x_train, y_train)
        return self

    def predict(self, x_test):
        y_pred = self.model.predict_proba(x_test)
        return y_pred
