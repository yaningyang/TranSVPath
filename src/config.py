import os


# features files path
bwFiles_root_dir = "../data/features_files/bw_files"        
gene_interval_file = '../data/gene_annotation/gene_interval.csv'
bedFiles_root_dir = "../data/features_files/bed_files/"
GDI_file_path = '../data/features_files/gene_files/GDI.txt'
Lof_file_path = '../data/features_files/gene_files/Lof.txt'
RVIS_file_path =  '../data/features_files/gene_files/RVIS.txt'
features_header_file = '../data/meta_data/features_header.txt'
features_header_depos_file = '../data/meta_data/features_header_exculde_pos.txt'
number_names_file = '../data/meta_data/NUM_COLUMNS.txt'

SV_TYPE = ['Deletion', 'Duplication', 'Insertion', 'Inversion', 'Microsatellite']

def get_bw_files():
    bwFiles = []
    for dirpath, _, filenames in os.walk(bwFiles_root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            bwFiles.append(file_path)
    return bwFiles

def get_bed_files():
    bedFiles = []
    for dirpath, _, filenames in os.walk(bedFiles_root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            bedFiles.append(file_path)
    return bedFiles


def convert_chrom_to_int(chrom):
    if chrom == "X":
        return 23
    elif chrom == "Y":
        return 24
    else:
        try:
            chrom_int = int(chrom)
            if 1 <= chrom_int <= 22:
                return chrom_int
        except ValueError:
            pass
    return None


def get_features_header(features_file):
    features_header = []
    with open(features_file, 'r') as file:
        for line in file:
            features_header.append(line.strip())
    return features_header

CSV_HEADER = get_features_header(features_header_file)
CSV_HEADER_DEPOS = get_features_header(features_header_depos_file)
NUM_COLUMNS_NAMES = get_features_header(number_names_file)

categorical_columns = [
        'GDI-all', 'GDI-Mendelian', 'GDI-Mendelian-AD', 'GDI-Mendelian-AR',
        'GDI-all-PID', 'GDI-PID-AD', 'GDI-PID-AR', 'GDI-all-cancer',
        'GDI-cancer-recessive', 'GDI-cancer-dominant', 'molecular_consequence'
]



# A dictionary of the categorical features and their vocabulary.

CATEGORICAL_FEATURES_WITH_VOCABULARY = {
                                        'GDI-all': ['High', 'Low', 'Medium'],
                                        'GDI-Mendelian': ['High', 'Low', 'Medium'],
                                        'GDI-Mendelian-AD': ['High', 'Low', 'Medium'],
                                        'GDI-Mendelian-AR': ['High', 'Low', 'Medium'],
                                        'GDI-all-PID': ['High', 'Low', 'Medium'],
                                        'GDI-PID-AD': ['High', 'Low', 'Medium'],
                                        'GDI-PID-AR': ['High', 'Low', 'Medium'],
                                        'GDI-all-cancer': ['High', 'Low', 'Medium'],
                                        'GDI-cancer-recessive': ['High', 'Low', 'Medium'],
                                        'GDI-cancer-dominant': ['High', 'Low', 'Medium'],
                                        'molecular_consequence': ['3_prime_UTR_variant',
                                                                    '5_prime_UTR_variant',
                                                                    'frameshift_variant',
                                                                    'genic_downstream_transcript_variant',
                                                                    'genic_upstream_transcript_variant',
                                                                    'inframe_deletion',
                                                                    'inframe_indel',
                                                                    'initiator_codon_variant',
                                                                    'intron_variant',
                                                                    'no_sequence_alteration',
                                                                    'non-coding_transcript_variant',
                                                                    'nonsense',
                                                                    'splice_acceptor_variant',
                                                                    'splice_donor_variant',
                                                                    'stop_lost',
                                                                    'frameshift_deletion',
                                                                    'stopgain',
                                                                    'startloss',
                                                                    'unknown',
                                                                    'nonframeshift_deletion',
                                                                    'stoploss',
                                                                    'nonsynonymous_SNV',
                                                                    'synonymous_SNV',
                                                                    'None']
                                }

    
VALUES_TO_REMOVE = {"start", "end"}

WEIGHT_COLUMN_NAME = "chrom"
CATEGORICAL_FEATURE_NAMES = categorical_columns
FEATURE_NAMES = NUM_COLUMNS_NAMES + CATEGORICAL_FEATURE_NAMES
COLUMN_DEFAULTS = [[0.0] if feature_name in NUM_COLUMNS_NAMES + [WEIGHT_COLUMN_NAME] else ["NA"] for feature_name in CSV_HEADER_DEPOS]
# The name of the target feature.
TARGET_FEATURE_NAME = "Label"
# A list of the labels of the target features.
TARGET_LABELS = ["Pathogenic", "Benign"]

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DROPOUT_RATE = 0.2
BATCH_SIZE = 256
NUM_EPOCHS = 100

NUM_TRANSFORMER_BLOCKS = 3  # Number of transformer blocks.
NUM_HEADS = 4  # Number of attention heads.
EMBEDDING_DIMS = 32  # Embedding dimensions of the categorical features.
MLP_HIDDEN_UNITS_FACTORS = [2, 1,]  # MLP hidden layer units, as factors of the number of inputs.
NUM_MLP_BLOCKS = 2  # Number of MLP blocks in the baseline model.

def convert_chrom_to_int(chrom):
    if chrom == "X":
        return 23
    elif chrom == "Y":
        return 24
    else:
        try:
            chrom_int = int(chrom)
            if 1 <= chrom_int <= 22:
                return chrom_int
        except ValueError:
            pass
    return None

def convert_chrom_to_str(chrom):
    if chrom == 23:
        return "X"
    elif chrom == 24:
        return "Y"
    else:
        try:
            if 1 <= chrom <= 22:
                chrom_str = str(chrom)
                return chrom_str
        except ValueError:
            pass
    return None