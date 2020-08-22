from pyspark.sql import SparkSession
from core.utils import Utils

class Spark:

    def load_csv(self, file_alias):
        file_path = Utils.get_file_name(file_alias)
        return self._spark.read.format("csv").options(header='true', inferSchema='true')\
            .load('data/raw/{}'.format(file_path))

    def __init__(self):
        spark = SparkSession.builder \
            .master("local") \
            .appName("ADNI Data Cleaning")

        self._spark = spark.getOrCreate()
