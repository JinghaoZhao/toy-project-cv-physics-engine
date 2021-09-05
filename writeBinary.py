from array import array
import numpy as np

output_file = open('tmp/data/faceLandmark.binary', 'wb')
landmarks = np.random.rand(468*3)
float_array = array('d', landmarks)
float_array.tofile(output_file)
output_file.close()

# input_file = open('file', 'rb')
# float_array = array('d')
# float_array.fromstring(input_file.read())
