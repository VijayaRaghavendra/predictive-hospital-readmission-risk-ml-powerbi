import pandas as pd

# Load your raw claims file (update the filename to match your exact downloaded file)
# The sep='|' is crucial because CMS files use pipes
claims_raw = pd.read_csv('data/raw/inpatient.csv', sep='|', low_memory=False)

# Save it as the standard filename the guide expects
claims_raw.to_csv('data/raw/inpatient_claims.csv', index=False)

print(f"Success! Processed {len(claims_raw)} claims into inpatient_claims.csv")