# Copyright 2023.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Processamento e criação da Feature Table."""

import argparse
import importlib
import logging
import sys
from typing import Callable, List, NoReturn, Optional

from databricks import feature_store
import pyspark
from pyspark import context
from pyspark.sql import session


_DataFrame = pyspark.sql.DataFrame

sc = context.SparkContext.getOrCreate()
spark = session.SparkSession(sc)


def set_up_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    
    
def create_and_write_fs(
    output_table_name: str,
    primary_keys: str,
    timestamp_keys: List[str],
    features_df: _DataFrame) -> NoReturn:
    """Cria e popula uma Feature Table.
    
    Args:
       output_table_name: Nome da Feature Table. 
       primary_keys: Chaves primárias a serem usadas na tabela.
       timestamp_keys: Coluna a serem usadas como timestamp.
       features_df: Uma instância de `spark.sql.DataFrame` com os dados.
    """
    table_name = output_table_name.split('.')[0]
    logging.info('Feature Table %s', table_name)
    spark.sql(f'CREATE DATABASE IF NOT EXISTS {table_name}')
    
    fs = feature_store.FeatureStoreClient()
    fs.create_table(
        name=output_table_name,
        primary_keys=primary_keys,
        timestamp_keys=timestamp_keys,
        df=features_df)
    fs.write_table(
        name=output_table_name,
        df=features_df,
        mode='merge')

    
def read_data(
    input_table_path: str,
    read_fn: Optional[Callable] = None) -> _DataFrame:
    """Lê os dados como Delta table.
    
    Args:
       input_table_path: Uma Delta table contendo dados de entrada. 
    """
    logging.info('Lendo dados de %s', input_table_path)
    if read_fn:
        return read_fn(input_table_path, spark)
    return spark.read.format('delta').load(input_table_path)
    

def generate_features(
    input_table_path: str,
    output_table_name: str,
    features_transform: str,
    primary_keys: List[str],
    timestamp_keys: List[str] = None,
    file_reader: Optional[str] = None,
    argv: Optional[List[str]] = None,
    **kwargs: str) -> NoReturn:
    """Gera e escreve features em uma tabela na Feature Store.
    
    Os argumentos passados na função de preparo de dados podem
      ser informados de duas formas. Caso sejam informados como
      argumentos nomeados o `argv` não é necessário.
    
    Args:
        input_table_path: Uma Delta table contendo dados de entrada. 
        output_table_name: Nome da Feature Table.
        features_transform: Nome do módulo que implementa a geração 
          das features.
        primary_keys: Chaves primárias a serem usadas na tabela.
        timestamp_keys: Colunas a serem usadas como timestamp.
        file_reader: Nome do módulo que implementa uma leitura de arquivos.
        argv: Uma lista de strings com os argumentos usados na 
          geração das features. 
        **kwargs: Argumentos nomeados usados na geração das features.
    """
    feature_mod = importlib.import_module(features_transform)
    compute_features_fn = getattr(feature_mod, 'compute_features_fn')
     
    if argv:
        update_flags = getattr(feature_mod, 'update_flags')
        kwargs = update_flags(argv)
        
    reader_fn = None
    if file_reader:
        reader_mod = importlib.import_module(file_reader)
        reader_fn = getattr(reader_mod, 'read_fn')
    raw_data = read_data(input_table_path, reader_fn)
    
    logging.info('Chamando função do módulo externo.')
    features_df = compute_features_fn(
        input_df=raw_data,
        **kwargs
    )
    create_and_write_fs(
        output_table_name=output_table_name,
        primary_keys=primary_keys,
        timestamp_keys=timestamp_keys,
        features_df=features_df)
    
    
def main(argv=None):
    set_up_logging()
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_table_path',
        help='Uma Delta table contendo dados de entrada.',
        required=True)
    parser.add_argument(
        '--output_table_name',
        help='Nome da Feature Table onde os dados serão escritos.',
        required=True)
    parser.add_argument(
        '--features_transform',
        help='Nome do módulo que implementa a geração das features.',
        required=True)
    parser.add_argument(
        '--file_reader',
        help='Nome do módulo que implementa uma leitura de arquivos.',
        default=None)
    parser.add_argument(
        '--primary_keys', 
        help='Colunas que serão usadas como chave primária na Feature Table, \
        separadas por vírgula.')
    parser.add_argument(
        '--timestamp_keys', 
        help='Colunas a serem usadas como timestamp na Feature Table.',
        default=None)
    
    known, unknown = parser.parse_known_args(argv)
    generate_features(
        input_table_path=known.input_table_path,
        output_table_name=known.output_table_name,
        features_transform=known.features_transform,
        primary_keys=known.primary_keys,
        timestamp_keys=known.timestamp_keys,
        file_reader=known.file_reader,
        argv=unknown)
    
if __name__ == '__main__':
    main()