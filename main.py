from core.utils import Utils
from data.pipeline import Pipeline
import numpy as np
import pandas as pd
from collections import OrderedDict

if __name__ == "__main__" :

    experiment_pipeline_configs = Utils.read_yml('pipeline_conf.yml')

    EXPERIMENTS = 21
    average_results = {}
    for experiment in range(1, EXPERIMENTS):
        random_state = np.random
        config = {''}

        pipeline = Pipeline(config, random_state)
        # Build the raw dataset and apply transformations
        data = pipeline.build_data()

        results = {}

        failed_pipelines = []
        counter = 1

        for config in experiment_pipeline_configs:
            print('Model is {}'.format(config))
            print('Percentage done is {} / {}'.format(counter, len(experiment_pipeline_configs)))
            model_pipeline_config = experiment_pipeline_configs[config]
            pipeline = Pipeline(model_pipeline_config, random_state)

            # Define imaging data and non-imaging data
            data_final = pipeline.build_final_dataset(data.copy())
            data_list_final = pipeline.build_data_list(data_final.copy())

            # Feature Selection
            data_feature_selection, features_keep = pipeline.feature_selection(data_list_final)

            # Run the pipeline
            metrics = pipeline.run_model(data_feature_selection, experiment)

            metrics['top_features'] = features_keep
            results[config] = metrics
            #print(results[config])
            counter = counter + 1

        results = pd.DataFrame(results)
        #print(results)
        results.to_csv('results/results_{}.csv'.format(experiment))
        pipeline.build_visualizations(results, experiment)

