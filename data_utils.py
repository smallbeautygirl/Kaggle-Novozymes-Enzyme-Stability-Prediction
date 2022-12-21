"""
4  SETUP AND HELPER FUNCTIONS
4.1 HELPER FUNCTIONS
"""


def get_mutation_info(_row, _wildtype="VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQ"
                                      "RVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGT"
                                      "NAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKAL"
                                      "GSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK"):
    terminology_map = {
        "replace": "substitution",
        "insert": "insertion",
        "delete": "deletion"}
    req_edits = Levenshtein.editops(_wildtype, _row["protein_sequence"])
    _row["n_edits"] = len(req_edits)

    if _row["n_edits"] == 0:
        _row["edit_type"] = _row["edit_idx"] = _row["wildtype_aa"] = _row["mutant_aa"] = pd.NA
    else:
        _row["edit_type"] = terminology_map[req_edits[0][0]]
        _row["edit_idx"] = req_edits[0][1]
        _row["wildtype_aa"] = _wildtype[_row["edit_idx"]]
        _row["mutant_aa"] = _row["protein_sequence"][req_edits[0]
                                                     [2]] if _row["edit_type"] != "deletion" else pd.NA
    return _row


def revert_to_wildtype(protein_sequence, edit_type,
                       edit_idx, wildtype_aa, mutant_aa):
    if pd.isna(edit_type):
        return protein_sequence
    elif edit_type != "insertion":
        new_wildtype_base = protein_sequence[:edit_idx]
        if edit_type == "deletion":
            new_wildtype = new_wildtype_base + \
                wildtype_aa + protein_sequence[edit_idx:]
        else:
            new_wildtype = new_wildtype_base + \
                wildtype_aa + protein_sequence[edit_idx + 1:]
    else:
        new_wildtype = protein_sequence[:edit_idx] + \
            wildtype_aa + protein_sequence[edit_idx:]
    return new_wildtype


def flatten_l_o_l(nested_list):
    """ Flatten a list of lists """
    return [item for sublist in nested_list for item in sublist]


def print_ln(symbol="-", line_len=110):
    print(symbol * line_len)

# Note mutation edit_idx is offset by 1 as many tools require it to be 1
# indexed to 0 indexed.


def create_mutation_txt_file(_test_df, filename="/kaggle/working/AF70_mutations.txt",
                             return_mutation_list=False, include_deletions=False):
    if return_mutation_list:
        mutation_list = []
    with open(filename, 'w') as f:
        for _, _row in _test_df[["protein_sequence", "edit_type",
                                 "edit_idx", "wildtype_aa", "mutant_aa"]].iterrows():
            if not include_deletions and (
                    pd.isna(_row["edit_type"]) or _row["edit_type"] == "deletion"):
                continue
            f.write(
                f'{_row["wildtype_aa"]+str(_row["edit_idx"]+1)+(_row["mutant_aa"] if not pd.isna(_row["mutant_aa"]) else "")}\n')
            if return_mutation_list:
                mutation_list.append(
                    f'{_row["wildtype_aa"]+str(_row["edit_idx"]+1)+(_row["mutant_aa"] if not pd.isna(_row["mutant_aa"]) else "")}')
    if return_mutation_list:
        return mutation_list


def create_wildtype_fasta_file(
        wildtype_sequence, filename="/kaggle/working/wildtype_af70.fasta"):
    with open(filename, 'w') as f:
        f.write(f">af70_wildtype\n{wildtype_sequence}")


def uniprot_id2seq(uniprot_id, _sleep=3):
    base_url = "http://www.uniprot.org/uniprot"
    full_url = os.path.join(base_url, str(uniprot_id) + ".fasta")
    _r = requests.post(full_url)
    if _r.status_code != 200:
        print(_r.status_code)
    if _sleep != -1:
        time.sleep(_sleep)
    return ''.join(_r.text.split("\n")[1:])
