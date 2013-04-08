import numpy as np
import scipy.io
import os.path

for file_name in [a_file for a_file in sorted(os.listdir('./')) if a_file[-4:] == '.dat']:
    print 'Processing %s' % file_name
    # Read file
    with open(file_name, 'r') as input_file:
        text = input_file.readlines()
    # Remove header
    text = [line for line in text if not line[0] == '@']
    # Convert labels into numbers
    IO_dict = {}
    new_text = []
    counter = 0
    for line in text:
        last_symbol = line.split(',')[-1]
        if not last_symbol in IO_dict:
            IO_dict[last_symbol] = '%d\n' % counter
            counter += 1
        new_text.append(','.join(line.split(',')[:-1] + [IO_dict[last_symbol]]))
    # Write to file
    with open(file_name, 'w') as output_file:
        output_file.write(''.join(new_text))
