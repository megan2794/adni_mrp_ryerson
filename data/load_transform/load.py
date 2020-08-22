from core.spark import Spark
from pyspark.sql import functions as F

class Load:

    def __init__(self):
        self.spark = Spark()
        self.data_raw = self.load_raw_data()
        self.feature_groupings = self.get_feature_definitions()

    def load_raw_data(self):

        study_df = self.spark.load_csv('studydata')
        study_df_final = self.spark.load_csv('studydatafinal')
        adni_merge = self.spark.load_csv('adnimerge')
        adni_all = self.spark.load_csv('adniall')

        study_df_final = study_df_final.withColumnRenamed('VisitMonth', 'VisCode') \
            .join(adni_merge.drop('update_stamp').drop('DX').drop('EXAMDATE').drop('age'),on=['RID', 'VisCode'],
                  how='left').withColumnRenamed('Diagnosis', 'DX')
        study_df_final = study_df_final.join(adni_all.drop('update_stamp').drop('Phase').drop('EXAMDATE'),
                                             on=['RID', 'VisCode'], how='left')

        #study_df = study_df.join(adni_merge.drop('update_stamp').drop('DX').drop('EXAMDATE'), on=['RID', 'VisCode'],
        #                         how='left')
        #study_df = study_df.join(adni_all.drop('update_stamp').drop('Phase').drop('EXAMDATE'),
        #                         on=['RID', 'VisCode'], how='left')

        #return study_df.withColumn('DX', F.when(F.col('DX') == 'CN', 0)
        #                     .when(F.col('DX') == 'SMC', 1).when(F.col('DX') == 'EMCI', 2)
        #                     .when(F.col('DX') == 'LMCI', 3).when(F.col('DX') == 'AD', 4))
        return study_df_final.withColumn('DX', F.when(F.col('DX') == 'CN', 0)
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
        return self.data_raw.select(list(set(features_list)))

    def load(self):

        clinical_data = self.get_feature_subset(['Cognitive Assessments', 'Neuropsychological Tests'])
        mri_data = self.get_feature_subset(['MRI Imaging'])
        pet_data = self.get_feature_subset(['PET Imaging'])
        sociodemographic_data = self.get_feature_subset(['Sociodemographic'])
        fluid_data = self.get_feature_subset(['Fluid Biomarkers'])

        return clinical_data, mri_data, pet_data, sociodemographic_data, fluid_data