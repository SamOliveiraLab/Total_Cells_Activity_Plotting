# In order to use this script you must have a nd2 file in a directory

## Dependencies 
- openCV
- numpy
- matplotlib
- imageio
- nd2
- nd2reader
- alive_progress

### Data folder structure
Given a input path directory, the final folder specified by `INPUT_PATH` should present the following structure:
```bash
Input directory 
├── 1.nd2
└── 2.nd2
```
note: there could be more than nd2 files

### Use the nd2_box notebook to find the parameters for detecting cell box and change these parameters before running the activityPlot.py scirpt:
example:
```python
# CONSTANTS 
FL_BLANK = 505.57
FL_CHERRY_BLANK = 510.57
CHERRY_RPU = 1151.74
YFP_RPU = 1117.03 

# Change these value according to nd2_box notebook
XY_CELL_HEIGHT = 910
XY_CELL_WIDTH = 640
XY_CELL_LIGHT_BAR_WIDTH = 10

LEFT_BOUND = 500
RIGHT_BOUND = 1200
```


## To run the script:
```bash
usage: python activityPlot.py -i 'INPUT_PATH' 

Convert nd2 files into graph, gifs, and csv

arguments:
  -i  Input_path: path to the directory the nd2csv files are located
```


