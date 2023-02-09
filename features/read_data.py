def read_fn(path, spark):
    return spark.read.load(
        path, format="csv",sep=";",
        inferSchema="true",header="true")