"""
3  IMPORTS
"""
import ast
import gc
import gzip
import hashlib
import io
import json
import math
import os
import pickle
import random
import re
import shutil
import string
import sys
import time
import urllib
import warnings
import zipfile
from collections import Counter
from datetime import datetime
from glob import glob
from io import StringIO
from zipfile import ZipFile

import Bio
import biopandas
import cv2
import imageio
import IPython
import Levenshtein
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
import seaborn as sns
import sklearn
import tensorflow as tf
import tifffile as tif
from Bio import SeqIO
from Bio.SubsMat import MatrixInfo
from biopandas.pdb import PandasPdb
from matplotlib import animation, rc
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from PIL import Image, ImageEnhance
from scipy import stats
from tqdm.notebook import tqdm

from data_utils import get_mutation_info

sigmoid_norm_factor = 3
print("\n... IMPORTS STARTING ...\n")
print("\n\tVERSION INFORMATION")
# Biology Specific Imports (You'll see why we need these later)
# from kaggle_datasets import KaggleDatasets

print(f"\t\t– BioPython VERSION: {Bio.__version__}")

# Built-In Imports (mostly don't worry about these)
# Visualization Imports (overkill)

print(f"\t\t– BioPandas VERSION: {biopandas.__version__}")
pdb = PandasPdb()

# Machine Learning and Data Science Imports (basics)
print(f"\t\t– TENSORFLOW VERSION: {tf.__version__}")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
print(f"\t\t– NUMPY VERSION: {np.__version__}")
print(f"\t\t– SKLEARN VERSION: {sklearn.__version__}")


tqdm.pandas()
Image.MAX_IMAGE_PIXELS = 5_000_000_000
print(f"\t\t– MATPLOTLIB VERSION: {matplotlib.__version__}")
rc('animation', html='jshtml')

print(pio.renderers)


def seed_it_all(seed=7):
    """ Attempt to be Reproducible """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_it_all()

print("\n\n... IMPORTS COMPLETE ...\n")


"""
4  SETUP AND HELPER FUNCTIONS
4.2 LOAD DATA - Much of this may not be needed.
"""
# Define the path to the root data directory
DATA_DIR = "data"

print("\n... BASIC DATA SETUP STARTING ...\n")
print("\n\n... LOAD TRAIN DATAFRAME FROM CSV FILE ...\n")
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
print(train_df)

print("\n\n... LOAD TEST DATAFRAME FROM CSV FILE ...\n")
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print(test_df)

print("\n\n... LOAD SAMPLE SUBMISSION DATAFRAME FROM CSV FILE ...\n")
ss_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
print(ss_df)

print("\n\n... LOAD ALPHAFOLD WILDTYPE STRUCTURE DATA FROM PDB FILE ...\n")
pdb_df = pdb.read_pdb(
    os.path.join(
        DATA_DIR,
        "wildtype_structure_prediction_af2.pdb"))

print("ATOM DATA...")
atom_df = pdb_df.df['ATOM']
print(atom_df)

print("\nHETATM DATA...")
hetatm_df = pdb_df.df['HETATM']
print(hetatm_df)

print("\nANISOU DATA...")
anisou_df = pdb_df.df['ANISOU']
print(anisou_df)

print("\nOTHERS DATA...")
others_df = pdb_df.df['OTHERS']
print(others_df)

print("\n\n... SAVING WILDTYPE AMINO ACID SEQUENCE...\n")
wildtype_aa = "VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK"
print(wildtype_aa)

print("\n\n... DEFINE AMINO ACID SHORTFORM DICTIONARY MAPPING...\n")
aa_map = dict(Alanine=("Ala", "A"), Arginine=("Arg", "R"), Asparagine=("Asn", "N"), Aspartic_Acid=("Asp", "D"),
              Cysteine=("Cys", "C"), Glutamic_Acid=("Glu", "E"), Glutamine=("Gln", "Q"), Glycine=("Gly", "G"),
              Histidine=("His", "H"), Isoleucine=("Ile", "I"), Leucine=("Leu", "L"), Lysine=("Lys", "K"),
              Methionine=("Met", "M"), Phenylalanine=("Phe", "F"), Proline=("Pro", "P"), Serine=("Ser", "S"),
              Threonine=("Thr", "T"), Tryptophan=("Trp", "W"), Tyrosine=("Tyr", "Y"), Valine=("Val", "V"))
n_aa = len(aa_map)
aa_chars_ordered = sorted([v[1] for v in aa_map.values()])
aa_long2tri = {k: v[0] for k, v in aa_map.items()}
aa_long2char = {k: v[1] for k, v in aa_map.items()}
aa_tri2long = {v: k for k, v in aa_long2tri.items()}
aa_char2long = {v: k for k, v in aa_long2char.items()}
aa_char2int = {_aa: i for i, _aa in enumerate(aa_chars_ordered)}
aa_int2char = {v: k for k, v in aa_char2int.items()}

# Get data source map
print("\n\n... DEFINE DATASOURCE DICTIONARY MAPPING...\n")
ds_str2int = {k: i for i, k in enumerate(train_df["data_source"].unique())}
ds_int2str = {v: k for k, v in ds_str2int.items()}

for k, v in aa_map.items():
    print(
        f"'{k}':\n\t3 LETTER ABBREVIATION --> '{v[0]}'\n\t1 LETTER ABBREVIATION --> '{v[1]}'\n")

print("\n\n... FOR FUN ... HERE IS THE ENTIRE WILDTYPE WITH FULL AMINO ACID NAMES (8 AA PER LINE) ...\n")
for i, _c in enumerate(wildtype_aa):
    print(f"'{aa_char2long[_c]}'", end=", ") if (
        i + 1) % 8 != 0 else print(f"{aa_char2long[_c]}", end=",\n")

print("\n\n... ADD COLUMNS FOR PROTEIN SEQUENCE LENGTH AND INDIVIDUAL AMINO ACID COUNTS/FRACTIONS ...\n")
train_df["n_AA"] = train_df["protein_sequence"].apply(len)
test_df["n_AA"] = test_df["protein_sequence"].apply(len)
for _aa_char in aa_chars_ordered:
    train_df[f"AA_{_aa_char}__count"] = train_df["protein_sequence"].apply(
        lambda x: x.count(_aa_char))
    train_df[f"AA_{_aa_char}__fraction"] = train_df[f"AA_{_aa_char}__count"] / \
        train_df["n_AA"]
    test_df[f"AA_{_aa_char}__count"] = test_df["protein_sequence"].apply(
        lambda x: x.count(_aa_char))
    test_df[f"AA_{_aa_char}__fraction"] = test_df[f"AA_{_aa_char}__count"] / test_df["n_AA"]

print("\n... ADD COLUMNS FOR DATA SOURCE ENUMERATION ...\n")
train_df["data_source_enum"] = train_df['data_source'].map(ds_str2int)
test_df["data_source_enum"] = test_df['data_source'].map(ds_str2int)

print("\n... DO TEMPORARY pH FIX BY SWAPPING pH & tm IF pH>14 ...\n")


def tmp_ph_fix(_row):
    if _row["pH"] > 14:
        print(f"\t--> pH WEIRDNESS EXISTS AT INDEX {_row.name}")
        _tmp = _row["pH"]
        _row["pH"] = _row["tm"]
        _row["tm"] = _tmp
        return _row
    else:
        return _row


print(f"\t--> DOES THE  STILL EXIST: {train_df['pH'].max()>14.0}")
train_df = train_df.progress_apply(tmp_ph_fix, axis=1)
test_df = test_df.progress_apply(tmp_ph_fix, axis=1)

print("\n\n\n... BASIC DATA SETUP FINISHED ...\n\n")


"""
5  STEP-BY-STEP WALKTHROUGH – PLLDT AND DDG
5.1 PREREQUISITE DATA FILES
"""

# 1.1 – Define the path to the model output text file (I'm using mine but
# you can use the original)
DEEPDDG_PRED_TXT_PATH = "data/own/wildtype_structure_prediction_af2.deepddg.ddg.txt"

# 1.2 – Read the txt file into a pandas dataframe specifying a space as the delimiter
#       --> This introduces some weirdness with the number of columns... so we have to drop some and rename them
#       --> We also drop the #chain column as it contains no usable information (all rows are the same --> A)
#       --> We also rename the columns to be more user friendly (note ddg stands for ΔΔG)
deepddg_pred_df = pd.read_table(
    DEEPDDG_PRED_TXT_PATH,
    sep=" ").drop(
        columns=[
            "#chain",
            "ddG",
            "is",
            "stable,",
            "is.1",
            "unstable)",
            "<0"])
deepddg_pred_df.columns = [
    "wildtype_aa",
    "residue_id",
    "mutant_aa",
    "ddg",
    "ddg_"]

# 1.3 – Coerce all the ddg values to be in the right column
deepddg_pred_df["ddg"] = deepddg_pred_df["ddg"].fillna(
    0.0) + deepddg_pred_df["ddg_"].fillna(0.0)
deepddg_pred_df = deepddg_pred_df.drop(columns=["ddg_"])

# 1.4 – Change edit location string name and change from 1-indexed to 0-indexed
deepddg_pred_df.loc[:, 'location'] = deepddg_pred_df["residue_id"] - 1
deepddg_pred_df = deepddg_pred_df.drop(columns=["residue_id"])

# 1.5 – Create a new column that contains the entire mutation as a string
#   --> This mutation string has the format   <wildtype_aa><location><mutant_aa>
deepddg_pred_df.loc[:, 'mutant_string'] = deepddg_pred_df["wildtype_aa"] + \
    deepddg_pred_df["location"].astype(str) + deepddg_pred_df["mutant_aa"]

# 1.6 – Display the newly created datafarme containing predictions (and
# describe float/int based columns)
print(deepddg_pred_df.describe().T)
print(deepddg_pred_df)

"""
5.2 UPDATE WILDTYPE ATOM DATAFRAME INFORMATION
"""

# 2.1 – Adjust the residue column to be 0-indexed instead of 1-indexed
atom_df['residue_number'] -= 1

# 2.2 – Create a mapping from residue to b-factor
#   --> b-factor is always the same for a particular residue
residue_to_bfactor_map = atom_df.groupby(
    'residue_number').b_factor.first().to_dict()

"""
5.3 UPDATE TEST DATAFRAME
"""
# 3.1 – Add mutation information about test set
#   --> Type of Mutation ['substitution'|'deletion'|'no_change'|'insertion'] (insertion never occurs)
#   --> Position of Mutation (Note this is now 0 indexed from the previous cell/section)
#   --> Wildtype Amino Acid (Single Letter Short Form)
#   --> Mutant Amino Acid (Single Letter Short Form – Will be NaN in 'deletion' mutations)
test_df = test_df.progress_apply(get_mutation_info, axis=1)

# 3.2 – Add b-factor to the test dataframe using previously created dictionary
test_df['b_factor'] = test_df["edit_idx"].map(residue_to_bfactor_map)

# 3.3 – Change edit_type from NaN to 'no_change'
#   --> this will allow the entire column to be a string as NaN is considered a float
test_df["edit_type"] = test_df["edit_type"].fillna("no_change")

# 3.4 – Change mutant_aa from NaN to '+' or '-' if edit_type is 'insertion' or 'deletion' respectively
#   --> this will allow the entire column to be a string as NaN is considered a float
test_df.loc[test_df['edit_type'] == 'deletion', 'mutant_aa'] = '-'
test_df.loc[test_df['edit_type'] == 'insertion', 'mutant_aa'] = '+'

# 3.5 – Create a new column that contains the entire mutation as a string
#   --> This mutation string has the format   <wildtype_aa><edit_idx><mutant_aa>
test_df.loc[:, 'mutant_string'] = test_df["wildtype_aa"] + \
    test_df["edit_idx"].astype(str) + test_df["mutant_aa"]
test_df[["seq_id", "edit_type", "edit_idx", "wildtype_aa",
         "mutant_aa", "b_factor", "mutant_string"]].head()

# 3.6 – Drop columns we don't need for this notebook
test_df = test_df[[_c for _c in test_df.columns if (
    ("__" not in _c) and ("data_source" not in _c))]]

# 3.7 – Display the updated dataframe (and describe float/int based columns)
print(test_df.describe().T)
print(test_df)

"""
5.4 COMBINE TEST DATA WITH DEEPDDG PREDICTIONS
"""
# 4.1 – Merge the two dataframes together
test_df = test_df.merge(
    deepddg_pred_df[['ddg', 'mutant_string']], on='mutant_string', how='left')

# 4.2 – Fill in the missing ddg values with predetermined values
#   --> We set the default value for deletion to be equivalent to the bottom quartile value
#       of all substitutions... this is because it is more deleterious than simple substitutions
#   --> The default no_change value is simply 0.0 because this is the wildtype
#   --> What I would have thought is better:
#            ----> DEFAULT__DELETION__DDG  = test_df[test_df["edit_type"]=="substitution"]["ddg"].quantile(q=0.25)
DEFAULT__DELETION__DDG = -0.25
DEFAULT__NO_CHANGE__DDG = 0.0   # THIS IS DIFFERENT THAN THE ORIGINAL NOTEBOOK
test_df.loc[test_df['edit_type'] == "deletion", 'ddg'] = DEFAULT__DELETION__DDG
test_df.loc[test_df['edit_type'] == "no_change",
            'ddg'] = DEFAULT__NO_CHANGE__DDG

# 4.3 – Display the updated dataframe (and describe float/int based columns)
print(test_df.describe().T)
print(test_df)

"""
5.5 PERFORM MATRIX SUBSTITUTION (BLOSUM-100)
"""

# 5.1 – Define a function to return the substitution matrix (backwards and
# forwards)


def get_sub_matrix(matrix_name="blosum100"):
    sub_matrix = getattr(MatrixInfo, matrix_name)
    sub_matrix.update({(k[1], k[0]): v for k, v in sub_matrix.items() if (
        k[1], k[0]) not in list(sub_matrix.keys())})
    return sub_matrix


sub_matrix = get_sub_matrix()

# 5.2 – Conduct matrix substitution
#   --> First we create a tuple that has the wildtype amino acid and the
#       mutant amino acid to access the substitution matrix
#   --> Second we access the substitution matrix and replace with the respective score
# and in cases where no respective score is found we mark it to be updated
# later
test_df["sub_matrix_tuple"] = test_df[[
    "wildtype_aa", "mutant_aa"]].apply(tuple, axis=1)
test_df["sub_matrix_score"] = test_df["sub_matrix_tuple"].progress_apply(
    lambda _mutant_tuple: sub_matrix.get(_mutant_tuple, "tbd"))

# 5.3 – Fill in the missing data with default values for now
#   --> We set the default value for matrix sub to be equivalent to the bottom quartile value
#       of all substitutions... this is because it is more deleterious than simple substitutions (larger difference)
#   --> The default no_change value is 1 higher than the max score because a higher score means more similarity
#DEFAULT__DELETION__MATRIXSCORE  = test_df[test_df["edit_type"]=="substitution"]["sub_matrix_score"].quantile(q=0.25)
DEFAULT__DELETION__MATRIXSCORE = -10.0
#DEFAULT__NO_CHANGE__MATRIXSCORE = test_df[test_df["edit_type"]=="substitution"]["sub_matrix_score"].max()+1.0
DEFAULT__NO_CHANGE__MATRIXSCORE = 0.0
test_df.loc[test_df['edit_type'] == "deletion",
            'sub_matrix_score'] = DEFAULT__DELETION__MATRIXSCORE
test_df.loc[test_df['edit_type'] == "no_change",
            'sub_matrix_score'] = DEFAULT__NO_CHANGE__MATRIXSCORE
test_df["sub_matrix_score"] = test_df["sub_matrix_score"].astype(float)

# 5.4 – Display the updated dataframe (and describe float/int based columns)
print(test_df.describe().T)
print(test_df)

"""
5.6 ADJUST THE MATRIX SUBSTITUTION AND B_FACTOR VALUES
"""
sub_scores = []
sub_mat = MatrixInfo.blosum100
for i in range(len(test_df)):
    mut_type = test_df.edit_type.values[i]
    if mut_type == 'substitution':
        try:
            sub_score = sub_mat[(test_df.old_aa.values[i],
                                 test_df.new_aa.values[i])]
        except KeyError:
            sub_score = sub_mat[(test_df.new_aa.values[i],
                                 test_df.old_aa.values[i])]
    elif mut_type == 'nothing':
        sub_score = 0
    else:
        sub_score = -10
    sub_scores.append(sub_score)

test_df['sub_score'] = sub_scores

test_df.loc[test_df['sub_score'] > 0, 'sub_score'] = 0
test_df['score_adj'] = [
    1 - (1 / (1 + np.exp(-x / sigmoid_norm_factor))) for x in sub_scores
]
test_df['b_factor_adj'] = test_df['b_factor'] * test_df['score_adj']

# 6.1 – If the flag is set, reduce all positive matrix substitution scores to 0
CAP_SUB_MATRIX_SCORE = True
CAP_ABOVE_VAL, CAP_VAL = 0.0, 0.0
if CAP_SUB_MATRIX_SCORE:
    test_df.loc[test_df['sub_matrix_score'] >
                CAP_ABOVE_VAL, 'sub_matrix_score'] = CAP_VAL

# 6.2 – Normalize the matrix substitution score with adjusted sigmoid function
SIGMOID_ADJUSTMENT_CONSTANT = 3


def sigmoid_w_adjustment(x, adjustment_factor=3.0):
    return 1 - (1 / (1 + np.exp(-x / adjustment_factor)))


test_df['sub_matrix_score_normalized'] = test_df['sub_matrix_score'].apply(
    lambda x: sigmoid_w_adjustment(x, SIGMOID_ADJUSTMENT_CONSTANT))
test_df['b_factor_matrix_score_adjusted'] = test_df['b_factor'] * \
    test_df['sub_matrix_score_normalized']

# 6.3 – Display the updated dataframe (and describe float/int based columns)
print(test_df.describe().T)
print(test_df)

"""
5.7 CALCULATE RANKS OF IMPORTANT COLUMNS
"""


# 7.1 – Assign ranks to data, dealing with ties appropriately.
test_df['ddg_rank'] = stats.rankdata(test_df['ddg'])
test_df['sub_matrix_score_rank'] = stats.rankdata(test_df['sub_matrix_score'])
test_df['b_factor_rank'] = stats.rankdata(-test_df['b_factor'])
test_df['b_factor_matrix_score_adjusted_rank'] = stats.rankdata(
    -test_df['b_factor_matrix_score_adjusted'])

# 7.2 – Display the updated dataframe (and describe float/int based columns)
print(test_df.describe().T)
print(test_df)
