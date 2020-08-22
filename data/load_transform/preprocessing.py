from core.spark import Spark
from pyspark.sql import functions as F
from core.utils import Utils

class Preprocessing:

    def get_raw_data(self):

        study_df = self.spark.load_csv('studydata')
        adni_merge = self.spark.load_csv('adnimerge')
        adni_all = self.spark.load_csv('adniall')

        study_df = study_df.join(adni_merge.drop('update_stamp').drop('DX').drop('EXAMDATE'), on=['RID', 'VisCode'],
                                 how='left')
        study_df = study_df.join(adni_all.drop('update_stamp').drop('Phase').drop('EXAMDATE'),
                                 on=['RID', 'VisCode'], how='left')

        return study_df.withColumn('DX', F.when(F.col('DX') == 'CN', 0)
                             .when(F.col('DX') == 'SMC', 1).when(F.col('DX') == 'EMCI', 2)
                             .when(F.col('DX') == 'LMCI', 3).when(F.col('DX') == 'AD', 4))

    def get_feature_definitions(self):

        feature_groupings_adnimerge = self.spark.load_csv('featuregroupings_adnimerge')
        feature_groupings_adniall = self.spark.load_csv('featuregroupings_adniall')
        return feature_groupings_adnimerge.unionAll(feature_groupings_adniall)

    def get_feature_subset(self, keep_group):
        features = self.feature_groupings.filter(F.col('Grouping').isin(keep_group)).select('Feature').collect()
        features_list = []
        for i in range(len(features)):
            features_list.append(features[i][0])
        features_list.append('DX')
        return self.data.select(list(set(features_list)))

    def clean_nulls(self, df, feature, rules):
        try:
            return df.withColumn(feature, F.when(F.col(feature).isin(rules['nullVals']), F.lit(None)) \
                                 .otherwise(F.col(feature)))
        except:
            return df

    def replace_nulls(self, df, feature, rules):
        try:
            # TODO: Look into more methods for handling null values
            if rules['nullHandling'] == 'remove':
                return df.filter((F.col(feature).isNotNull()) | ~(F.col(feature) == 'null'))
            elif rules['nullHandling'] == 'mean_imputation':
                value = df.select(F.col(feature)).agg({feature: 'mean'}).collect()[0][0]
                return df.withColumn(feature, F.when(F.col(feature).isNull(), value).otherwise(F.col(feature)))
        except:
            pass
        return df

    def replace_values(self, df, feature, rules):
        try:
            for replacement in rules['replaceVals']:
                df = df.withColumn(feature, F.regexp_replace(feature, replacement[0], (replacement[1])))
        except:
            pass
        return df

    def clean_data_tye(self, df, feature, rules):
        try:
            return df.withColumn(feature, F.col(feature).cast(rules['fieldType']))
        except:
            return df

    def normalize_based_on_other_feature(self, df, feature, rules):
        try:
            normalize_on = rules['normalizeOtherFt']
            # TODO - check decimal precision here
            return df.withColumn(feature, F.col(feature) / F.col(normalize_on))
        except:
            return df

    def clean_data(self, df, cleansing_rules):

        for feature in cleansing_rules:

            # Drop the column if required
            try:
                if cleansing_rules[feature]['drop'] == 'true':
                    df = df.drop(feature)
            except:
                pass

            # Clean Null Values
            df = self.clean_nulls(df, feature, cleansing_rules[feature])
            # Handle Null Values
            df = self.replace_nulls(df, feature, cleansing_rules[feature])
            # Replace values in the column
            df = self.replace_values(df, feature, cleansing_rules[feature])
            # Clean up the data type
            df = self.clean_data_tye(df, feature, cleansing_rules[feature])
            # Clean up the data type
            df = self.normalize_based_on_other_feature(df, feature, cleansing_rules[feature])

        return df

    def clean(self):
        clinical_data = self.get_feature_subset(['Cognitive Assessments', 'Neuropsychological Tests'])
        self.clinical_data = self.clean_data(clinical_data, Utils.get_cleansing_rules('clinical'))
        mri_data = self.get_feature_subset(['MRI Imaging'])
        self.mri_data = self.clean_data(mri_data, Utils.get_cleansing_rules('mri'))
        pet_data = self.get_feature_subset(['PET Imaging'])
        self.pet_data = self.clean_data(pet_data, Utils.get_cleansing_rules('pet'))
        soc_data = self.get_feature_subset(['Sociodemographic'])
        self.soc_data = self.clean_data(soc_data, Utils.get_cleansing_rules('sociodemographic'))
        fluid_data = self.get_feature_subset(['Fluid Biomarkers'])
        self.fluid_data = self.clean_data(fluid_data, Utils.get_cleansing_rules('fluid_biomarkers'))

    def __init__(self):
        self.spark = Spark()
        self.data = self.get_raw_data()
        self.feature_groupings = self.get_feature_definitions()
        self.clean_data = self.clean()
