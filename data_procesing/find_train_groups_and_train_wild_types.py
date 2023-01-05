from collections import defaultdict
from operator import itemgetter
from statistics import StatisticsError, mode

import numpy as np
import pandas as pd

"""
Load Train
"""
train = pd.read_csv('data/updated_train.csv')
print('Train shape:', train.shape )

## Visually show the data is fixed. Interesting that there are still the same number of unique data sources after the "bad data" was removed.
#  train['data_source'].nunique() -> 統計 "data_source" 列中不同值得個數，不包刮 null 值
print(train['pH'].min(), train['pH'].max(), train['data_source'].nunique())

print(train.head())

"""
Find wildtype candidates for each length of protein string
"""

train['seq_len'] = train.protein_sequence.str.len() # protein_sequence 長度
print(train.head())
vc = train.seq_len.value_counts() # The resulting object will be in descending order so that the first element is the most frequently-occurring element. Excludes NA values by default.

# vc (Series)
# value, values count
print(vc.head())

# INSERTION DELETION THRESHOLD
D_THRESHOLD = 1
# MIN GROUP SIZE
MIN_GROUP_SIZE = 5

def max_item_count(seq):
    d = defaultdict(int) # default值以一個list()方法產生
    for item in seq:
        d[item] += 1
    return max(d.items(), key=itemgetter(1))

def get_wildtype(proteins, is_retry=False):
    if not is_retry:
        ## try to get the mode, the simpler algorithm
        wildtype = []
        try:
            for i in range(len(proteins.iloc[0])):
                wildtype.append(mode([p[i] for p in proteins]))
            return ''.join(wildtype)
        except StatisticsError:
            pass

   ## Either failed mode above, or this is a retry because the resulting wildtype didn't actually fit enough proteins
    ##
    ## Two sequences with single mutation from the same wildtype are no more than 2 points different.
    ## Therefore, at least 1/3rd length consecutive string must match. Find max counts of starts, middles, and ends
    ## This technically isn't a guaranteed or precise algorithm, but it is fast and effective,
    ##   based on comparison with more precise grouping methods.
    k = len(proteins.iloc[0])//3  # //運算子不管除數、被除數的型別為何，回傳的都是無條件捨去的結果。如果都是整數，就回傳整數，如果有任一為浮點數，則回傳浮點數：
    starts = [p[:k] for p in proteins]
    middles = [p[k:2*k] for p in proteins]
    ends = [p[-k:] for p in proteins]
    ## get the most common substring, and the count of that substring
    start = max_item_count(starts)
    middle = max_item_count(middles)
    end = max_item_count(ends)
    ## reduce the proteins to the ones that match the most common substring
    if (start[1] >= middle[1]) and (start[1] >= end[1]) and (start[1] >= MIN_GROUP_SIZE):
        proteins = [p for p in proteins if p[:k] == start[0]]
        assert(start[1] == len(proteins))
    elif (middle[1] >= end[1]) and (middle[1] >= MIN_GROUP_SIZE):
        proteins = [p for p in proteins if p[k:2*k] == middle[0]]
        assert(middle[1] == len(proteins))
    elif end[1] >= MIN_GROUP_SIZE:
        proteins = [p for p in proteins if p[-k:] == end[0]]
        assert(end[1] == len(proteins))
    else:
        return ''
    ## use the reduced list to find the entire wildtype
    wildtype = []
    try:
        for i in range(len(proteins[0])):
            wildtype.append(mode([p[i] for p in proteins]))
        return ''.join(wildtype)
    except StatisticsError:
        return ""


train['group'] = -1
train['wildtype'] = ''
grp = 0

for k in range(len(vc)):
    if vc.iloc[k] < MIN_GROUP_SIZE:
        break
    c = vc.index[k]
    print(f'rows={vc.iloc[k]}, k:{k}, protein length:{c}')
    is_retry = False
    # SUBSET OF TRAIN DATA WITH SAME PROTEIN LENGTH (not enough deletions to matter for step 1, finding the wildtype)
    tmp = train.loc[(train.seq_len==c)&(train.group==-1)]

    ## It is possible that the same length protein string might have multiple wildtypes in the raw data, keep searching until we've found all of them
    while len(tmp) >= MIN_GROUP_SIZE:
        if len(tmp)<=1: break
        # Ignore Levenstein distance, which is overkill
        # Directly attempt to find wildtype
        # Drop duplicates for wildtype guesstimation
        proteins = tmp.protein_sequence.drop_duplicates()

        # Create most likely wildtype
        wildtype = get_wildtype(proteins, is_retry=is_retry)
        if wildtype == '':
            break

        # SUBSET OF TRAIN DATA WITH SAME PROTEIN LENGTH PLUS MINUS D_THRESHOLD
        tmp = train.loc[(train.seq_len>=c-D_THRESHOLD)&(train.seq_len<=c+D_THRESHOLD)&(train.group==-1)]
        for idx in tmp.index:
            p = train.loc[idx, 'protein_sequence']
            half = c//2
            ## Use fast method to guess that it is only a single point mutation away. Later we double check and actually count number of mutations.
            if (wildtype[:half] == p[:half]) or (wildtype[-half:] == p[-half:]):
                train.loc[idx,'group'] = grp
                train.loc[idx,'wildtype'] = wildtype
        if len(train.loc[train.group==grp]) >= MIN_GROUP_SIZE:
            print(f"{train.loc[(train.group==grp)].shape[0]}: Group {grp} results")
            grp += 1
            is_retry = False
        else:
            train.loc[idx,'group'] = -1
            train.loc[idx,'wildtype'] = ''
            ## to avoid an infinite loop, break out if we've already failed last time
            if is_retry:
                break
            is_retry = True

        # Get ready for next loop
        tmp = train.loc[(train.seq_len==c)&(train.group==-1)]

print(f'grp: {grp}\n')

"""
Display Groups
"""

def argsort(seq, reverse=False):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)

groups = [0] * grp
for k in range(grp):
    groups[k] = len(train.loc[train.group==k])

groupCount = 0
rowCount = 0
for k in argsort(groups, reverse=True):
    if train.loc[train.group==k].shape[0] == 0:
        continue
    proteins = train.loc[train.group==k, "protein_sequence"]
    wildtype = train.loc[train.group==k, "wildtype"].values[0]

    ## no insertions in the dataset, that I've found.
    ## Handle deletions by adding a '-' symbol in the correct place
    for i in range(len(proteins)):
        if len(proteins.iloc[i]) < len(proteins.iloc[0]):
            if proteins.iloc[i] == wildtype[:-1]:
                proteins.iloc[i] = proteins.iloc[i] + "-"
            else:
                for j in range(len(proteins.iloc[i])):
                    if proteins.iloc[i][j] != wildtype[j]:
                        proteins.iloc[i] = proteins.iloc[i][:j-1] + "-" + proteins.iloc[i][j:]
                        break
        assert(len(proteins.iloc[i]) == len(proteins.iloc[0]))


    ## Ungroup those proteins.
    ungroup = []
    for p in proteins:
        mut = 0
        for j in range(len(wildtype)):
            if p[j] != wildtype[j]:
                mut += 1
        if mut > 1:
            if p not in ungroup:
                ungroup.append(p)
    for p in ungroup:
        train.loc[train.protein_sequence==p, 'group'] = -1
        train.loc[train.protein_sequence==p, 'wildtype'] = ''
    ## Remove entire group if it is now smaller than the min group size
    if train.loc[train.group==k].shape[0] < MIN_GROUP_SIZE:
        train.loc[train.group==k, 'wildtype'] = ''
        train.loc[train.group==k, 'group'] = -1
        continue

    ## Print a line for every group, and a bunch of stats for the first few groups
    print(f'{k}: {train.loc[train.group==k].shape[0]}')
    if groupCount < 5:
        print( train.loc[train.group==k] )
        for c in train.columns:
            print(c, train.loc[train.group==k, c].nunique() + train.loc[train.group==k, c].isnull().values.any())
        print(wildtype)
        print("")
    groupCount += 1
    rowCount += train.loc[train.group==k].shape[0]

print(groupCount, rowCount)

## Re-number groups from largest to smallest
groups = [0] * grp
for k in range(grp):
    groups[k] = len(train.loc[train.group==k])

n = 10000
for k in argsort(groups, reverse=True):
    train.loc[train.group==k, "group"] = n
    n += 1

train.loc[train.group>=10000, "group"] = train.loc[train.group>=10000, "group"] - 10000
train.loc[train.group==-1, "group"] = 1000
train = train.sort_values(axis=0, by=['group'], kind='mergesort').reset_index(drop=True)
train.loc[train.group==1000, "group"] = -1

# train = train.drop('x',axis=1)
train_wildtype_groups = train.loc[train.wildtype != '']
train_no_wildtype = train.loc[train.wildtype == '']
print(train_wildtype_groups.shape)
print(train_no_wildtype.shape)

train_wildtype_groups.to_csv('data/train_wildtype_groups.csv',index=False)
train_wildtype_groups.head()

train_no_wildtype.to_csv('data/train_no_wildtype.csv',index=False)
train_no_wildtype.head()