# ADNI MRP - Megan Parker
MRP Code for ADNI Dataset

This project contains a framework for feature extraction and prediction of Altzheimer's Disease (AD) using a subset of the ADNI Dataset.


Running the project:
```
conda create -n env python=3.7
conda activate env
pip install -r requirements.txt
export MPLBACKEND="agg"
python3 main.py
```

The project supports the following feature selection algorithms:
  1. PCA [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
  2. LDA [sklearn.discriminant_analysis.LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
  3. Relieff [relieff implementation from pypi](https://pypi.org/project/ReliefF/)
  4. Information Gain [from sklearn.feature_selection.mutual_info_classif](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)
  5. Mrmr [pymrmr implementation from pypi](https://pypi.org/project/pymrmr/)
  6. Random Forest Feature Importance [sklearn.ensemble.ExtraTreesClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)

The project uses GMM to determine the optimal number of features selected in each feature selection algorithm [sklearn.mixture.GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)

Project Directory:
```
conf
| cleansing_rules.yml
| cleansing_rules_preclean.yml
| data.yml
core
| spark.py
| utils.py
data
| clean
  | preprocessing.py
| featureselection
  | feature_selection.py
  | gmm.py
| predict
  | simple_classifiers.py
| raw
notebooks
  | ADNI_DataCleansing.ipynb
  | ADNI_visualizations.ipynb
  | Data_Cleansing_Library.ipynb
  | data_exploration.ipynb
  | data_exploration_aftercleansing.ipynb
  | feature_extraction.ipynb
  | feature_extraction_tests.ipynb

/notebooks contains files which can be used for Exploratory Data Analysis:
  - ADNI.ipynb reads the data from the ADNI subset and outputs feature summaries including histograms and scatter plots
  - Data_Cleansing_Library.ipynb contains various data cleansing functions

/data contains the classes used for data cleaning, feature selection and building the model
raw data is held in data/raw/ and is excluded from the git repo
