"""
As has been pointed out, there are some data issues in the training data. A file has been added to the Data page which contains the rows that should not be used due to data quality issues (2409 rows, with all features marked as NaN), as well as the rows where the pH and tm were transposed (25 rows, with corrected features in this dataset).
"""

import os

import numpy as np
import pandas as pd

DATA_DIR = "data"

"""
Setup For Fixing
"""


train_df = pd.read_csv(os.path.join(DATA_DIR,"train.csv"))
train_updates_df = pd.read_csv(os.path.join(DATA_DIR,"train_updates_20220929.csv"))

# Identify which sequence ids need to have the tm and pH values changed and create a dictionary mapping
seqid_2_phtm_update_map = train_updates_df[~pd.isna(train_updates_df["pH"])].groupby("seq_id")[["pH", "tm"]].first().to_dict("index")

# Identify the sequence ids that will be dropped due to data quality issues
bad_seqids = train_updates_df[pd.isna(train_updates_df["pH"])].seq_id.to_list()

"""
Demonstrate the Problem
"""

# Data quality issue rows
print("\n... EXAMPLES OF 10 ROWS WITH DATA QUALITY ISSUE ...\n")
print(train_df[train_df.seq_id.isin(bad_seqids)].head(10))

print("\n... EXAMPLES OF 10 ROWS WHERE pH & tm HAVE BEEN SWAPPED ERRONEOUSLY ...\n")
print(train_df[train_df.pH>14.0].head(10))
### OR ###
# display(train_df[train_df.seq_id.isin(list(seqid_2_phtm_update_map.keys()))].head(10))

"""
Do the Fixing
"""

# Drop useless all NaN rows
train_df = train_df[~train_df["seq_id"].isin(bad_seqids)].reset_index(drop=True)

# Correct tm-->pH swap
def fix_tm_ph(_row, update_map):
    update_vals = update_map.get(_row["seq_id"], None)
    if update_vals is not None:
        _row["tm"] = update_vals["tm"]
        _row["pH"] = update_vals["pH"]
    return _row
train_df = train_df.apply(lambda x: fix_tm_ph(x, seqid_2_phtm_update_map), axis=1)

print("\n... WE CAN'T CHECK FOR THE BAD DATA ROWS BUT WE CAN CHECK FOR BROKEN pH/tm VALUES ...\n")
print(train_df[train_df.pH>14.0].head(10)) # This should yield an empty dataframe

# Save to disk
train_df.to_csv(os.path.join(DATA_DIR,"updated_train.csv"), index=False)

"""
Wrap this all up in a function so we can just get on with our lives...

"""
# Will take 3-5 seconds to run
def load_fixed_train_df(original_train_file_path="/kaggle/input/novozymes-enzyme-stability-prediction/train.csv",
                        update_file_path="/kaggle/input/novozymes-enzyme-stability-prediction/train_updates_20220929.csv",
                        was_fixed_col=False):
    def _fix_tm_ph(_row, update_map):
        update_vals = update_map.get(_row["seq_id"], None)
        if update_vals is not None:
            _row["tm"] = update_vals["tm"]
            _row["pH"] = update_vals["pH"]
        return _row

    # Load dataframes
    _df = pd.read_csv(original_train_file_path)
    _updates_df = pd.read_csv(update_file_path)

    # Identify which sequence ids need to have the tm and pH values changed and create a dictionary mapping
    seqid_2_phtm_update_map = _updates_df[~pd.isna(_updates_df["pH"])].groupby("seq_id")[["pH", "tm"]].first().to_dict("index")

    # Identify the sequence ids that will be dropped due to data quality issues
    bad_seqids = _updates_df[pd.isna(_updates_df["pH"])]["seq_id"].to_list()

    # Fix bad sequence ids
    _df = _df[~_df["seq_id"].isin(bad_seqids)].reset_index(drop=True)

    # Fix pH and tm swaparoo
    _df = _df.apply(lambda x: _fix_tm_ph(x, seqid_2_phtm_update_map), axis=1)

    # Add in a bool to track if a row was fixed or not (tm/ph swap will look the same as bad data)
    if was_fixed_col: _df["was_fixed"] = _df["seq_id"].isin(bad_seqids+list(seqid_2_phtm_update_map.keys()))

    return _df


print(load_fixed_train_df())
print(load_fixed_train_df(was_fixed_col=True))