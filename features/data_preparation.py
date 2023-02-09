from pyspark.sql import functions as F


def add_id_column(df, column_name):
    columns = df.columns
    new_df = df.withColumn(column_name, F.monotonically_increasing_id())
    return new_df[[column_name] + columns]
 
    
def rename_columns(df):
    renamed_df = df
    for col in df.columns:
        renamed_df = renamed_df.withColumnRenamed(col, col.replace(' ', '_'))
    return renamed_df


def compute_features_fn(input_df):
    renamed_df = rename_columns(input_df)
    df = add_id_column(renamed_df, 'wine_id')
    features_df = df.drop('quality')
    return features_df