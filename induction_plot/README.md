# This script converts csv files into graph with induction input and saves it as sgv

## In order to use this script you must have a csv file in the format of 7 columns 
### Example:
| Time (m) | Levels | RPU |	Error | aTc | IPTG | M9-Plain |
|----------|--------|-----|-------|-----|------|----------| 
|0	| 519.7814734	|0.023241869    |0.000401951| 0 | 0 | 1 | 
|15	| 521.3368924	|0.025785648	|0.0006051  | 0 | 0 | 1 |
|30	|523.9431084	|0.030047932	|0.000533227| 0 | 1 | 0 |
|45	|526.7044082	|0.034563844	|0.000878189| 0 | 1 | 0 |
|60	|530.7842578	|0.041236152	|0.001233598| 1 | 0 | 0 |
|75	|533.7470604	|0.046081609	|0.001511749| 1 | 0 | 0 |
|90	|537.3384941	|0.051955147	|0.001773445| 1 | 0 | 0 |

# `IMPORTANT`: 
There must be no blank space, replace black space with 0 as seen in the example above


## Dependencies 
- numpy
- matplotlib

### Data folder structure
Given a input path directory, the final folder specified by `INPUT_PATH` should present the following structure:
```bash
Input directory 
└── yfp.csv
```

## To run the script for yfp:
```bash
usage: python Induction_Plot.py -i 'INPUT_PATH' 

Convert csv file into graph with induction input and saves it as sgv

arguments:
  -i  Input_path: path to the directory the yfp csv file are located
```

## To run the script for m-cherry:
```bash
usage: python Induction_Plot_Red.py -i 'INPUT_PATH' 

Convert csv file into graph with induction input and saves it as sgv

arguments:
  -i  Input_path: path to the directory the m-cherry csv file are located
```