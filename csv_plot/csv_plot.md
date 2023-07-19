# In order to use this script you must have a csv file in the format of 4 columns 
### Example:
| Time (m) | Levels | RPU |	Error |
|----------|--------|-----|-------|
|0	| 519.7814734	|0.023241869    |0.000401951|
|15	| 521.3368924	|0.025785648	|0.0006051| 
|30	|523.9431084	|0.030047932	|0.000533227|
|45	|526.7044082	|0.034563844	|0.000878189|
|60	|530.7842578	|0.041236152	|0.001233598|
|75	|533.7470604	|0.046081609	|0.001511749|
|90	|537.3384941	|0.051955147	|0.001773445|


## Dependencies 
- openCV
- numpy
- matplotlib
- imageio

### Data folder structure
Given a input path directory, the final folder specified by `INPUT_PATH` should present the following structure:
```bash
Input directory 
├── m_cherry.csv
└── yfp.csv
```

## To run the script:
```bash
usage: python csv_plot.py -i 'INPUT_PATH' 

Convert edited csv files into graph and gifs

arguments:
  -i  Input_path: path to the directory the csv files are located
```


