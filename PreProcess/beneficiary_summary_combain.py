import pandas as pd
import glob
import os

# 1. Set the correct path
path = 'data/raw/' 
all_files = glob.glob(os.path.join(path, "*Beneficiary_Summary*.csv"))

if not all_files:
    print("No files found! Check if they are in data/raw/")
else:
    li = []

    for filename in all_files:
        # FIX 1: Add sep='|' because CMS files use pipes
        df = pd.read_csv(filename, sep='|', index_col=None, header=0)
        li.append(df)

    # Stack all years together
    full_bene = pd.concat(li, axis=0, ignore_index=True)

    # FIX 2: Use 'BENE_ID' (matches your uploaded file) instead of 'DESYNPUF_ID'
    # We sort by Year to keep the most recent patient info
    if 'BENE_ENROLLMT_REF_YR' in full_bene.columns:
        full_bene = full_bene.sort_values('BENE_ENROLLMT_REF_YR', ascending=False)
    
    full_bene = full_bene.drop_duplicates(subset='BENE_ID', keep='first')

    # Save as the single file the project expects (standard CSV)
    full_bene.to_csv('data/raw/beneficiary_summary.csv', index=False)
    print(f"Success! Created beneficiary_summary.csv with {len(full_bene)} unique patients.")