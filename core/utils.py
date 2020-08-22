import yaml
import matplotlib.pyplot as plt

class Utils:

    @staticmethod
    def get_file_name(data):
        conf_file = open('conf/data.yml')
        conf_file = yaml.safe_load(conf_file)
        return conf_file[data]

    @staticmethod
    def plot_scatter(x, y, title):
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap='rainbow')
        plt.title(title)
        plt.show()

    @staticmethod
    def read_yml(file_name):
        conf_file = open('conf/{}'.format(file_name))
        return yaml.safe_load(conf_file)
