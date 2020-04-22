## Bash Script for the Parser

### Description
The file *run.sh* in the *system* folder launches the python scripts that build a PCFG Grammar and can parse sentences using a probabilistic CYK algorithm.

It takes a text file of sentences of spaced tokens as input, and returns the parsed grammatical result with grammatical structure and Part-of-speech tags.

### Example
Input file:

    Je mange une pomme .
    Le cheval est dans les champs .
    Une salade de fruits .

Output file:

    ( (SENT (VN (CLS Je) (V mange)) (NP (DET une) (NC pomme)) (PONCT .)))
    ( (SENT (NP (DET Le) (NC cheval)) (VN (V est)) (PP (P dans) (NP (DET les) (NC champs))) (PONCT .)))
    ( (SENT (NP (DET Une) (NC salade) (PP (P de) (NP (NC fruits)))) (PONCT .)))


### Arguments

It takes 2 arguments as input:

 - *input_filename* which is the name of the file containing the raw sentences to parse.
 - *output_filename* which is the name of the file in which we will store the parsed results.

The arguments will be asked clearly when running *run.sh*:

    Name of input file with sentences to parse ? e.g file.txt
    input.txt
    Name of output file to store the results ? e.g output.txt
    result.txt
