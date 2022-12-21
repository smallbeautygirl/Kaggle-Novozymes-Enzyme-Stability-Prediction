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
from biopandas.pdb import PandasPdb
from matplotlib import animation, rc
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from PIL import Image, ImageEnhance
from tqdm.notebook import tqdm

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
4.2 LOAD DATA
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
