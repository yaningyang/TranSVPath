import sys
sys.path.append('.')

import csv
import numpy as np
import pandas as pd
import os
import multiprocessing
import argparse
from sklearn import preprocessing
from functools import partial
from tabtransformer import get_dataset_from_csv_2
from datetime import datetime
from config import *
from generate_features import generate_features
from read_vcf import read_vcf
import tensorflow as tf
from tensorflow.keras import layers
import pickle
from config import *

target_label_lookup_2 = layers.StringLookup(vocabulary=TARGET_LABELS, mask_token=None, num_oov_indices=1)

def prepare_example_2(features, target):
    target_index = target_label_lookup_2(target)
    weights = features.pop(WEIGHT_COLUMN_NAME)
    return features, target_index, weights

def get_dataset_from_csv_2(csv_file_path, batch_size=128, shuffle=False):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER_DEPOS,
        column_defaults=COLUMN_DEFAULTS,
        label_name=TARGET_FEATURE_NAME,
        num_epochs=1,
        header=False,
        na_value="?",
        shuffle=shuffle,
    ).map(prepare_example_2, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return dataset.cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TranSVPath, a Transformer-based deep learning framework for predicting the pathogenicity of '
                                     'deletions, insertions, inversions, tandem duplications, and microsatellite variants.')
    parser.add_argument('-f', '--vcf', required=True, help='Input VCF file, in which the type of variation should be clearly indicated in the INFO field '
                                                            '(SVTYPE=Duplication or Insertion or Microsatellite or Inversion or Deletion), '
                                                            'and the end position of the variation should also be given (END=...). '
                                                            'In addition, this file should be annotated with Annovar software to obtain molecular consequences and gene symbols')
    parser.add_argument('-m', '--model', required=True, help='A path for loading the saved models')
    parser.add_argument('-p', '--processes', type=int, default=8, help='Number of processes for parallel extraction of feature values')
    parser.add_argument('-o', '--output',  help='A txt file for output pathogenicity scores')

    args = parser.parse_args()
    vcf_file = args.vcf
    num_processes = args.processes
    model_path = args.model
    output_path = args.output

    SVs = read_vcf(vcf_file)
    SVs_positions = {var_type :[] for var_type in SVs.keys()}
    bwFiles = get_bw_files()
    bedFiles = get_bed_files()
    features_dir = 'features'
    for var_type, variations in SVs.items():
        partial_func = partial(generate_features, gene_interval_file, bwFiles, bedFiles, GDI_file_path, Lof_file_path, RVIS_file_path)

        print(f'Generating features for {var_type}.')
        start_time = datetime.now()
        pool = multiprocessing.Pool(processes=num_processes)
        variations_features = pool.map(partial_func, variations)
        pool.close()
        pool.join()

        end_time = datetime.now()
        time_diff = end_time - start_time
        seconds = time_diff.seconds

        print(f"{num_processes} processes time: {seconds} seconds for {var_type} variation")

        for sv_feature in variations_features:
            if sv_feature:
                SVs_positions[var_type].append(sv_feature[:3])        
        
        df_features = pd.DataFrame(variations_features, columns=CSV_HEADER)
        df_features['chrom'] = df_features['chrom'].apply(convert_chrom_to_int)
        df_features = df_features.drop(columns=VALUES_TO_REMOVE)
        
        if not os.path.exists(features_dir):
            os.makedirs(features_dir)
            
        df_features.to_csv(os.path.join(features_dir, var_type+'_features.csv'), index=False, header=None)
        print('Saved features to {}'.format(os.path.join(features_dir, var_type+'_features.csv')))

    if os.path.exists(output_path):
        os.remove(output_path)
    
    COLUMN_DEFAULTS = [[0.0] if feature_name in NUM_COLUMNS_NAMES + [WEIGHT_COLUMN_NAME] else ["NA"] for feature_name in CSV_HEADER_DEPOS]
        
    for var_type in SVs:
        if var_type in ['Duplication', 'Deletion', 'Insertion', 'Microsatellite']:
            continue

        data_tf = get_dataset_from_csv_2(os.path.join(features_dir, var_type+'_features.csv'), batch_size=128)
        model_dir = os.path.join(model_path, var_type + '.pkl') 
        print(f'Loading {var_type} model..')
        with open(model_dir, 'rb') as f:
            loaded_model = pickle.load(f)
        y_pred_prob = loaded_model.predict(data_tf)
        sv_score = [float(f"{x[0]:.4f}") for x in y_pred_prob]
        
        for i in range(len(SVs_positions[var_type])):
            result = SVs_positions[var_type][i] + [var_type, sv_score[i]]
            result = '\t'.join(map(str, result)) + '\n'
            with open(output_path, 'a') as out_file:
                out_file.write(result)