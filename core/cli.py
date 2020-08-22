import click
from data.models.model import Model


@click.command()
@click.option('--feature_selection', default='pca', help='Feature Selection Method')
@click.option('--classification', default='randomforest', help='Classification Method')
@click.option('--pipeline', default='1', help='Pipeline Configuration')
def build(feature_selection, classification, pipeline):
    '''Clean data and run ML'''
    model = Model(feature_selection, classification, pipeline)
    Model.run(model)

if __name__ == '__main__':
    build()
