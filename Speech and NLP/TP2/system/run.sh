#!/bin/bash
echo Name of input file with sentences to parse ? e.g file.txt
read input_filename

echo Name of output file to store the results ? e.g output.txt
read output_filename

python3 main.py --input_filename=$input_filename --output_filename=$output_filename
