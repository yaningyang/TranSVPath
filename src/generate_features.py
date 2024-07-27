import csv
import pyBigWig
import numpy as np
import pandas as pd
import os
import pysam
import time
import multiprocessing
import argparse
from sklearn import preprocessing
from functools import partial
import random

def calculate_overlap(row, query_start, query_end):
    overlap_start = max(row['start'], query_start)
    overlap_end = min(row['end'], query_end)
    return overlap_start, overlap_end

def get_bw_data_features(bwFiles, query_chr, query_start, query_end, interval_file):
    region_list = ["gene", "exon", "CDS", "start_codon", "stop_codon", "selenocysteine", "transcript"]
    score_list = ["max", "min", "mean"]
    overlap_scores = {region: {score: [] for score in score_list} for region in region_list}
    bwfiles_scores = dict.fromkeys(bwFiles, overlap_scores.copy())
    score_dict = {"max":np.max, "min": np.min, "mean": np.mean}
    #print(overlap_scores)
    if 'chr' not in query_chr:
            query_chr = 'chr' + query_chr
    df = pd.read_csv(interval_file)
    overlapping_genes = df[(df['chr'] == query_chr) &  (df['start'] <= query_end) & (df['end'] >= query_start)]
    if overlapping_genes.empty:
        # print("overlapping_genes is empty")
        overlap_scores = {region: [0]*len(score_list) for region in region_list}
    else:
        #print(len(overlapping_genes))
        results = []
        for _, row in overlapping_genes.iterrows():
            overlap_start, overlap_end = calculate_overlap(row, query_start, query_end)
            results.append((row['gene'], row['type'], overlap_start, overlap_end))
        
        for bwfile in bwFiles:
            bw = pyBigWig.open(bwfile)
            for gene, region, overlap_start, overlap_end in results:
                try:                    
                    for score_index, score_type in zip([0, 1, 2], score_list):
                        if query_chr in bw.chroms() and overlap_start < overlap_end < bw.chroms()[query_chr]:
                            score = bw.stats(query_chr, overlap_start, overlap_end, type=score_type)
                            #print(score)
                            overlap_scores[region][score_list[score_index]].append(score if score !=[None] else [0])
                    
                except:
                    #overlap_scores = {region: [0]*len(score_list) for region in region_list}
                    pass
            bwfiles_scores[bwfile] = overlap_scores
            overlap_scores = {region: {score: [] for score in score_list} for region in region_list}
            bw.close()
      
    roadmap_final_result = []
    for bwfile, bwvalues in bwfiles_scores.items():
        for region, region_values in bwvalues.items():
            for k, v in region_values.items():
                flat_values = [x for y in v for x in y]
                tmp = score_dict[k](flat_values) if flat_values else 0
                roadmap_final_result.append(tmp)

    return roadmap_final_result


def get_bed_data_features(bedFiles, query_chr, query_start, query_end):
    overlapValueList = []
    if 'chr' not in str(query_chr):
        query_chr = 'chr' + query_chr
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    # Prevent multi-process conflicts
    randomA = random.randint(1, 10000)
    randomB = random.randint(10000, 99999)

    tmpFileName = 'tmp/' + query_chr + '_' + str(query_start) + '_' + str(query_end) + str(randomA) + '_' + str(randomB)
    with open(tmpFileName, 'w+') as writeTmp:
        writeTmp.write(query_chr + '\t' + str(query_start) + '\t' + str(query_end))
    for bedfile in bedFiles:
        intersectionFileName = tmpFileName + '_' + bedfile.split('/')[-1] + '_ints.tsv'
        bedtoolsCommand = 'bedtools intersect -a ' + tmpFileName + ' -b ' + bedfile + ' -wao > ' + intersectionFileName
        os.system(bedtoolsCommand)
        intersectDict = []
        with open(intersectionFileName, 'r') as intersect:
            d = {}
            d['filename'] = bedfile
            for line in intersect:
                lineList = line.split()
                overlapResult = lineList[-1]
                if (query_chr, query_start, query_end) in d:
                    d[(query_chr, query_start, query_end)] += int(overlapResult)
                else:
                    d[(query_chr, query_start, query_end)] = int(overlapResult)
            avgResult = d[(query_chr, query_start, query_end)] / (int(query_end) - int(query_start))
            overlapValueList.append(avgResult)
        deleteInterSectTmp = 'rm ' + intersectionFileName
        os.system(deleteInterSectTmp)
    deleteTmpBed = 'rm ' + tmpFileName
    os.system(deleteTmpBed)
    return overlapValueList

def calculate_gene_GDI_values(filename, gene_list):
    value1_list = []
    value2_list = []
    other_values_list = []
    
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        headers = next(reader)  # Skip the header row
        
        for row in reader:
            gene = row[0]
            if gene in gene_list:
                value1_list.append(float(row[1]))
                value2_list.append(float(row[2]))
                other_values_list.append(row[3:])
    
    max_value1 = max(value1_list) if value1_list else 0
    max_value2 = max(value2_list) if value2_list else 0
    
    # Transpose other_values_list to work with columns
    transposed_other_values = list(zip(*other_values_list))
    
    max_other_values = []
    for column in transposed_other_values:
        if 'High' in column:
            max_other_values.append('High')
        elif 'Medium' in column:
            max_other_values.append('Medium')
        else:
            max_other_values.append('Low')
    if not max_other_values:
        max_other_values = ['Low']*10
    return [max_value1] + [max_value2] + max_other_values


def calculate_gene_LOF_values(file_name, gene_list):
    lof_values = []

    with open(file_name, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        headers = next(reader)  # Skip the header row

        gene_index = headers.index('Gene')
        lof_index = headers.index('LoFtool_percentile')

        for row in reader:
            gene = row[gene_index]
            if gene in gene_list:
                lof_values.append(float(row[lof_index]))

    if not lof_values:
        return [0]

    max_value = max(lof_values)

    return [max_value]


def calculate_gene_RVIS_values(filename, gene_list):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    data = [line.split() for line in lines[1:]]
    df = pd.DataFrame(data, columns=['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5', 'Value1', 'Value2'])
    
    df['Value1'] = df['Value1'].astype(float)
    df['Value2'] = df['Value2'].astype(float)
    
    value1_list = []
    value2_list = []
    for gene in gene_list:
        mask = df.apply(lambda row: gene in row.values[:5], axis=1)
        value1_list.extend(df[mask]['Value1'].values)
        value2_list.extend(df[mask]['Value2'].values)
    
    if value1_list:
        max_value1 = max(value1_list)
    else:
        max_value1 = 0

    if value2_list:
        max_value2 = max(value2_list)
    else:
        max_value2 = 0

    return [max_value1, max_value2]






def generate_features(gene_interval_file, bwFiles, bedFiles, GDI_file_path, Lof_file_path, RVIS_file_path, sv_record):
    sv_items = { "chrom":0, "start":1, "end":2, "mc":3, "gene":4, "clnsig":5}
    #label_map = {"Pathogenic": 1, "Benign": 0}
    #print(sv_record)
    chrom = sv_record[sv_items["chrom"]]
    start = sv_record[sv_items["start"]]
    end   = sv_record[sv_items["end"]]
    length = end - start
    mole_consequence  = sv_record[sv_items["mc"]]
    gene  = sv_record[sv_items["gene"]]

    
    bw_features = get_bw_data_features(bwFiles, chrom, start, end, gene_interval_file)
    bed_features = get_bed_data_features(bedFiles, chrom, start, end)
    
    Lof_features = calculate_gene_LOF_values(Lof_file_path, gene)
    RVIS_features = calculate_gene_RVIS_values(RVIS_file_path, gene)
    GDI_features = calculate_gene_GDI_values(GDI_file_path, gene)
    
    record_features = [chrom, start, end, length] + bw_features + bed_features + Lof_features + RVIS_features + GDI_features + [mole_consequence] + ['X']
    return record_features