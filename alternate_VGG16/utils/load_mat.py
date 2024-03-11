import scipy.io

# Load the files
dmos_data = scipy.io.loadmat('../data/LIVE_release2/databaserelease2/dmos.mat')
refnames_data = scipy.io.loadmat('../data/LIVE_release2/databaserelease2/refnames_all.mat')

# Access the DMOS scores and reference names
dmos_scores = dmos_data['dmos']
reference_names = refnames_data['refnames_all']

print("DMOS SCORES ------------------------------")
print(dmos_scores)
print("REFERENCE NAMES --------------------------")
print(reference_names)
