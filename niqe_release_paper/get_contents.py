import os
from scipy.io import loadmat

# Load the MATLAB .mat file
mat_contents = loadmat(os.path.join('', 'modelparameters.mat'))

# Specify the path for the output text file
output_file_path = 'modelparameters_contents.txt'

# Open the text file in write mode
with open(output_file_path, 'w') as output_file:
    # Print the keys to see what variables are in the file
    print("Keys:", mat_contents.keys(), file=output_file)

    # Access specific variables by their names and print them
    print("\nmu_prisparam:", mat_contents['mu_prisparam'], file=output_file)
    print("\ncov_prisparam:", mat_contents['cov_prisparam'], file=output_file)

# Notify the user that the contents have been saved to the text file
print(f"MATLAB file contents saved to {output_file_path}")
