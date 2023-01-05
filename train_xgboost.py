"""
XGBoost - 5000 Mutations 200 PDB Files
"""

# DEFINE WHAT TO TRAIN WITH (and KFOLD VALIDATE) VERSUS HOLDOUT VALIDATE WITH
# ADD WORDS "kaggle.csv", "jin_tm.csv", "jin_train.csv", "jin_test.csv" to lists below
# IF YOU ADD MORE DATASETS, ADD THOSE WORDS TOO

KFOLD_SOURCES = ['jin_tm.csv','jin_train.csv','jin_test.csv']
HOLDOUT_SOURCES = ['kaggle.csv']

# IF WILD TYPE GROUP HAS FEWER THAN THIS MANY MUTATION ROWS REMOVE THEM
EXCLUDE_CT_UNDER = 25

# IF WE TRAIN WITH ALPHA FOLD'S PDBS WE MUST INFER WITH "PLDDT = TRUE"
# KAGGLE.CSV USES ALPHA FOLD PDB, SO SET BELOW TO TRUE WHEN TRAIN WITH KAGGLE.CSV
# JIN.CSV EXTERNAL DATA USES PROTEIN DATA BANK, SO SET BELOW TO FALSE WITH JIN DATA
USE_PLDDT_INFER = False

# IF WE WISH TO TRAIN WITH MIXTURE OF ALPHA FOLD AND PROTEIN DATA BANK PDB FILES
# THEN WE CAN EXCLUDE B_COLUMN AND THEN THERE IS NO PROBLEM
USE_B_COLUMN = False

VER = 17

"""
Download 3 External Mutation CSV
"""
import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, rankdata, spearmanr

data_folder = 'data'
download_data_folder = os.path.join(data_folder,"downloaded_csv")

pd.set_option('display.max_columns', 500)
# os.system('wget https://raw.githubusercontent.com/JinyuanSun/mutation-stability-data/main/train.csv')
# os.system('wget https://raw.githubusercontent.com/JinyuanSun/mutation-stability-data/main/test.csv')
# os.system('wget https://raw.githubusercontent.com/JinyuanSun/mutation-stability-data/main/tm.csv')
# os.system(f'mkdir {download_data_folder}; mv *csv {download_data_folder}')


df = pd.read_csv(os.path.join(download_data_folder,'train.csv'))
df = df.iloc[:,1:]
print('Downloaded train shape', df.shape )
df.head()

df2 = pd.read_csv(os.path.join(download_data_folder,'test.csv'))
df2 = df2.iloc[:,1:]
print('Downloaded test shape', df2.shape )
df2.head()

df3 = pd.read_csv(os.path.join(download_data_folder,'tm.csv'))
df3 = df3.iloc[:,1:]
print('Downloaded tm shape', df3.shape )
df3.head()


"""
Transform Kaggle Train Data into Mutation CSV
"""
# https://www.kaggle.com/code/roberthatch/novo-train-data-contains-wildtype-groups/notebook
kaggle = pd.read_csv(os.path.join(data_folder,'novo-train-data-contains-wildtype-groups/train_wildtype_groups.csv'))
print('Before processing Robert dataframe shape:', kaggle.shape )
print(kaggle.head())

kaggle['id'] = kaggle.data_source.astype('str') + '_' + kaggle.pH.astype('str') + '_' + kaggle.group.astype('str')
kaggle['ct'] = kaggle.groupby('id').id.transform('count')
kaggle['n'] = kaggle.groupby('id').tm.transform('nunique')
kaggle = kaggle.loc[kaggle.n>1]
kaggle = kaggle.sort_values(['group','ct'],ascending=[True,False])
KEEP = kaggle.groupby('group').id.agg('first').values
kaggle = kaggle.loc[kaggle.id.isin(KEEP)]

def find_mut(row):
    mut = row.protein_sequence
    seq = row.wildtype
    same = True
    for i,(x,y) in enumerate(zip(seq,mut)):
        if x!=y:
            same = False
            break
    if not same:
        row['WT'] = seq[i]
        row['position'] = i+1
        row['MUT'] = mut[i]
    else:
        row['WT'] = 'X'
        row['position'] = -1
        row['MUT'] = 'X'
    return row

grp = [f'GP{g:02d}' for g in kaggle.group.values]
kaggle['PDB'] = grp
kaggle = kaggle.apply(find_mut,axis=1)
kaggle = kaggle.loc[kaggle.position!=-1]
kaggle['base'] = kaggle.groupby('group').tm.transform('mean')
kaggle['dTm'] = kaggle.tm - kaggle.base
kaggle = kaggle.rename({'wildtype':'sequence','protein_sequence':'mutant_seq'},axis=1)
COLS = ['PDB','WT','position','MUT','dTm','sequence','mutant_seq']
kaggle = kaggle[COLS]

# https://www.kaggle.com/datasets/shlomoron/train-wildtypes-af
alphafold = pd.read_csv(os.path.join(data_folder,'train-wildtypes-af/alpha_fold_df.csv'))
dd = {}
for s in kaggle.sequence.unique():
    tmp = alphafold.loc[alphafold.af2_sequence==s,'af2id']
    if len(tmp)>0: c = tmp.values[0].split(':')[1]
    else: c = np.nan
    dd[s] = c

kaggle['CIF'] = kaggle.sequence.map(dd)
kaggle = kaggle.loc[kaggle.CIF.notnull()].reset_index(drop=True)
kaggle.to_csv('kaggle_train.csv',index=False)
print('After processing Robert dataframe shape:', kaggle.shape )
print(kaggle.head())

"""
Combine 4 CSV Files
"""
df['source'] = 'jin_train.csv'
df['dTm'] = np.nan
df['CIF'] = None
df2['source'] = 'jin_test.csv'
df2['dTm'] = np.nan
df2['CIF'] = None

df3 = df3.loc[~df3.PDB.isin(['1RX4', '2LZM', '3MBP'])].copy()
df3['source'] = 'jin_tm.csv'
df3['ddG'] = np.nan
df3['CIF'] = None
df3 = df3.rename({'WT':'wildtype','MUT':'mutation'},axis=1)
df3 = df3[df.columns]

kaggle['source'] = 'kaggle.csv'
kaggle['ddG'] = np.nan
kaggle = kaggle.rename({'WT':'wildtype','MUT':'mutation'},axis=1)
kaggle = kaggle[df.columns]

df = pd.concat([df,df2,df3,kaggle],axis=0,ignore_index=True)
del df2, df3, kaggle
print('Combined data shape',df.shape)
df.to_csv(f'all_train_data_v{VER}.csv',index=False)
df = df.loc[df.source.isin(KFOLD_SOURCES+HOLDOUT_SOURCES)]
print('Kfold plus Holdout shape',df.shape)
df = df.sort_values(['PDB','position']).reset_index(drop=True)
print(df.head())


"""
Download 200 PDB Files
"""
print('There are',df.PDB.nunique(),'PDB files to download')

# THE FOLLOWING PROTEINS SEQUENCES CANNOT BE ALIGNED BETWEEN CSV AND PDB FILE (not sure why)
bad = [f for f in df.PDB.unique() if len(f)>4]
bad += ['1LVE', '2IMM', '2RPN', '1BKS', '1BLC', '1D5G', '1KDX', '1OTR', '3BN0', '3D3B', '3HHR', '3O39']
bad += ['3BDC','1AMQ','1X0J','1TPK','1GLM','1RHG','3DVI','1RN1','1QGV']
bad += ['1SVX','4E5K']
print(f'We will ignore mutations from {len(bad)} PDB files')

# os.system(f'mkdir {os.path.join(data_folder,"downloaded_pdb")}')
# for p in [f for f in df.PDB.unique() if f not in bad]:
#     if p[:2]=='GP': continue # skip kaggle CIF
#     os.system(f'cd downloaded_pdb; wget https://files.rcsb.org/download/{p}.pdb')
