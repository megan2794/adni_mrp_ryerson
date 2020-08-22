import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


class Visualizations:

    def __init__(self, results, experiment):
        self.results = results
        self.experiment_num = experiment

    def build_whisker_plot(self):
        df_acc = pd.DataFrame({})
        df_auc = pd.DataFrame({})

        for experiment in self.results:
            keys = self.results[experiment].keys()
            keys_auc = [k for k in keys if 'auc' in k]
            keys_auc.remove('auc')
            keys_acc = [k for k in keys if 'acc' in k]
            keys_acc.remove('accuracy')

            tmp = []
            for k in keys_acc:
                tmp.append(self.results[experiment][k])
            df_acc[experiment] = tmp
            tmp = []
            for k in keys_auc:
                tmp.append(self.results[experiment][k])
            df_auc[experiment] = tmp

        plt.figure(figsize=(25, 10))
        df_acc.boxplot()
        plt.title('Categorical Accuracy for Cross Validation Folds')
        plt.xticks(rotation=10)
        plt.savefig('results/box_acc_{}.png'.format(self.experiment_num))
        plt.clf()

        plt.figure(figsize=(25, 10))
        df_auc.boxplot()
        plt.title('AUC for Cross Validation Folds')
        plt.xticks(rotation=10)
        plt.savefig('results/box_auc_{}.png'.format(self.experiment_num))
        plt.clf()

    def build_scatter_plot(self):

        x = []
        y1, y2 = [], []
        labels = []
        counter = 0
        for experiment in self.results:
            y1.append(self.results[experiment]['accuracy'])
            y2.append(self.results[experiment]['auc'])
            x.append(counter)
            labels.append(experiment)
            counter = counter + 1
        plt.figure(figsize=(25, 10))
        plt.scatter(x, y1)
        plt.scatter(x, y2)
        plt.legend(('Accuracy', 'AUC'))
        plt.xticks(x, labels)
        plt.title('Scatter Plot of AUC and Categorical Accuracy')
        plt.xlabel("Experiment")
        plt.ylabel("Metric")
        plt.xticks(rotation=10)
        plt.savefig('results/scatter_metrics_{}.png'.format(self.experiment_num))
