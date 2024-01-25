import oct2py

# Start Octave
# Specify the path to the Octave executable
octave_executable = 'D:\\APPS\\PROGRAMMING\\OCTAVE\\Octave-8.4.0\\mingw64\\bin\\octave-cli.exe'
oc = oct2py.Oct2Py(executable=octave_executable)

# Load template model
template_model = oc.load('templatemodel.mat')
mu_prisparam = template_model['templateModel'][0]
cov_prisparam = template_model['templateModel'][1]
mean_of_sample_data = template_model['templateModel'][2]
principle_vectors = template_model['templateModel'][3]

# Specify the path to your image file
image_path = 'parrots.bmp'

# Call computequality function
metric_value = oc.computequality(image_path, mu_prisparam, cov_prisparam, principle_vectors, mean_of_sample_data)

# Print the computed quality score
print(f'Quality Score: {metric_value}')

# Close Octave
oc.exit()
