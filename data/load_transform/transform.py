from core.spark import Spark
from core.utils import Utils
from pyspark.sql import functions as F


class Transform:

    def __init__(self, clinical_data, mri_data, pet_data, sociodemographic_data, fluid_data):
        self.spark = Spark()
        self.clinical_data = []
        self.clinical_data, self.mri_data, self.pet_data, self.sociodemographic_data, self.fluid_data = \
            clinical_data, mri_data, pet_data, sociodemographic_data, fluid_data

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
                if replacement[0] == 'null':
                    df = df.withColumn(feature, F.when(F.col(feature).isNull(), replacement[1])
                                       .otherwise(F.col(feature)))
                else:
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

        cleansing_rules = Utils.read_yml('cleansing_rules.yml')
        self.clinical_data = self.clean_data(self.clinical_data, cleansing_rules['clinical'])
        self.mri_data = self.clean_data(self.mri_data, cleansing_rules['mri'])
        self.pet_data = self.clean_data(self.pet_data, cleansing_rules['pet'])
        self.sociodemographic_data = self.clean_data(self.sociodemographic_data, cleansing_rules['sociodemographic'])
        self.fluid_data = self.clean_data(self.fluid_data, cleansing_rules['fluid_biomarkers'])

        return self.clinical_data, self.mri_data, self.pet_data, self.sociodemographic_data, self.fluid_data
