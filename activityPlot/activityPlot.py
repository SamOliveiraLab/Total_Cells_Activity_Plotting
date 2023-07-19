import nd2
import csv 
import os
import argparse
import math
import re
import numpy as np
import warnings
import cv2 as cv
import imageio.v2 as imageio
from nd2reader import ND2Reader
from matplotlib import pyplot as plt
from alive_progress import alive_bar

# known warning so ignore
warnings.filterwarnings("ignore")

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

def get_edges_coord(my_array):

    CONT_HEIGHT = XY_CELL_HEIGHT
    CONT_WIDTH = XY_CELL_WIDTH
    LIGHT_WIDTH = XY_CELL_LIGHT_BAR_WIDTH
    
    left_bound = LEFT_BOUND
    right_bound = RIGHT_BOUND

    row = my_array[955, left_bound:right_bound]
    col = my_array[:100, 750]

    sobel_row = cv.Sobel(row, cv.CV_64F, 0, 1, ksize=3)
    sobel_col = cv.Sobel(col, cv.CV_64F, 0, 1, ksize=3)

    sobel_row = np.abs(sobel_row).argmax()
    sobel_col = np.abs(sobel_col).argmax()

    top = sobel_col + 2
    bottom = top + CONT_HEIGHT
    
    if sobel_row > 20:
        right = sobel_row + left_bound - LIGHT_WIDTH
        left = right - CONT_WIDTH
    else:
        left = sobel_row + left_bound + LIGHT_WIDTH
        right = left + CONT_WIDTH - 100

    
    return left, right, top, bottom

def make_dirs(mult_experiment, dir_path):
    if mult_experiment:
        if not os.path.exists(os.path.join(dir_path, f'm-cherry_results_1')):
            os.makedirs(os.path.join(dir_path, f'm-cherry_results_1'))
        if not os.path.exists(os.path.join(dir_path, 'm-cherry_results_2')):
            os.makedirs(os.path.join(dir_path, 'm-cherry_results_2'))
        if not os.path.exists(os.path.join(dir_path, f'YFP_results_1')):
                os.makedirs(os.path.join(dir_path, f'YFP_results_1'))
        if not os.path.exists(os.path.join(dir_path, 'YFP_results_2')):
                os.makedirs(os.path.join(dir_path, 'YFP_results_2'))
    else:
        if not os.path.exists(os.path.join(dir_path, f'm-cherry_results')):
            os.makedirs(os.path.join(dir_path, f'm-cherry_results'))
        if not os.path.exists(os.path.join(dir_path, f'YFP_results')):
                os.makedirs(os.path.join(dir_path, f'YFP_results'))

def make_csv(dir_path, filename, levels, RPU, errors, time):

    data = np.transpose(np.array([time, levels, RPU, errors]))

    with open(os.path.join(dir_path, filename) , 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time (m)', 'Levels', 'RPU', 'Error'])
        writer.writerows(data)

def make_gif(input_path):
    imgs_path = []
    for path in os.listdir(input_path):
        # check if its a file and is a file type nd2
        if path.endswith('.png'):
            imgs_path.append(path)

    imgs = []
    for path in sorted(imgs_path, key=lambda f: int(re.sub('\D', '', f))):
        imgs.append(cv.imread(os.path.join(input_path,path))[...,::-1])

    imageio.mimsave(os.path.join(input_path, "mc_cell_graph.gif"), imgs)

def plot_activity(levels, timeSeries, errors, figure, title, out_dir, out_name, y_label, color, legend):
    plt.figure(figure)

    # convert time into hours from minutes 
    timeSeries = np.around((np.divide(timeSeries, 60)), decimals=2)

    plt.title(title)
    plt.plot(np.asarray(timeSeries), np.asarray(levels), color=color);
    plt.xlabel('Time (h)')
    plt.ylabel(y_label)
    plt.fill_between(
            np.asarray(timeSeries), 
            np.asarray(levels) - np.asarray(errors), 
            np.asarray(levels) + np.asarray(errors),
            color="#ededed",
            alpha=0.5
        )
    plt.legend([legend])
    plt.ylim(ymin=0, ymax=np.asarray(levels).max() + 0.5) 
    plt.savefig(os.path.join(out_dir, out_name)); 

def validate_prompt_int(prompt, min_range, max_range):
    while True:
        try:
            num = int(input(prompt))

            if (num >= min_range and num <= max_range):
                return num

            print(f'Please enter a number between {min_range} and {max_range}')
        except:
            print('Please enter a number')

def validate_prompt(prompt):
    while True:
        input_res = input(f'\n{prompt} [y/n]: ')

        if input_res == 'y' or input_res == 'Y':
            return True
        elif input_res == 'n' or input_res == 'N':
            return False
        else:
            print('\nPlease enter y or n')

def main(args):
    dir_path = f"{args.input_path}"
    # nd2 files in given directory path
    res = []
    for path in os.listdir(dir_path):
        # check if its a file and is a file type nd2
        if path.endswith('.nd2') and os.path.isfile(os.path.join(dir_path, path)):
            res.append(path)

    # sort by name
    res = sorted(res, key=lambda f: int(re.sub('\D', '', f))) 
    print(f"Found {len(res)} nd2 files in '{dir_path}'\n")
    for i, files in enumerate(res):
        print(f"\t{i + 1}) {files}")

    # check if nd2 exists 
    if len(res) == 0:
        print('No nd2 file(s) in given directory')
        return

    # check if the order is correct by user 
    if not validate_prompt("The files will be processed in this order, do you wish to proceed?"):
        print('Please rename the files and try again')
        return  
    
    time_gap = []

    if len(res) > 1:
        for i in range(len(res) - 1):
            time_gap.append(validate_prompt_int(f'\nIn mintues, what is the time gap between {res[i]} and {res[i + 1]}?: ', 0, 1000))

    time_gap.reverse()

    # Order is correct  
    nd2_files = [] 
    for path in res:
        nd2_files.append(nd2.imread(os.path.join(dir_path, path), dask=True))

    # How many experiment:
    mult_experiment = validate_prompt(f'Does this ND2 files contain 2 experiments?')
    
    XY2_start = np.Infinity

    if mult_experiment:
        num_time_series, num_XY, num_channels, h, w = nd2_files[0].shape
        print(f'\nTime Series: {num_time_series}, Number of XY: {num_XY}, Number of channels: {num_channels}, Image size: {h} x {w}')
        XY2_start = validate_prompt_int(f'\nAt which XY does the 2nd experiment start? {(1, num_XY)}: ', 1, num_XY)

    # Ask for the title of the experiment
    title = input('What is the name of the experiment? ')
    
    if mult_experiment:
        title_2 = input('What is the name of the 2nd experiment? ')

    # make directories to store plots
    make_dirs(mult_experiment, dir_path)

    # initial variable 
    yfp_levels = []
    yfp_RPUs = []
    yfp_error = []

    yfp_levels_2 = []
    yfp_RPUs_2 = []
    yfp_error_2 = []

    m_cherry_levels = []
    m_cherry_RPUs = []
    m_cherry_error = []

    m_cherry_levels_2 = []
    m_cherry_error_2 = []
    m_cherry_RPUs_2 = []
    
    timeSeries = []

    count = 0

    for k, filename in enumerate(res):
        # my_array = nd2.imread(os.path.join(dir_path, filename))
        my_array = ND2Reader(os.path.join(dir_path, filename))
        # TESTING SINGLE FILE
        # num_time_series, num_XY, num_channels, h, w = nd2_files[k].shape

        num_time_series = my_array.sizes['t']
        num_XY = my_array.sizes['v']
        num_channels = my_array.sizes['c']
        h = my_array.sizes['y']
        w = my_array.sizes['x']

        if mult_experiment:
            num_XY_2 = num_XY - XY2_start + 1 
            num_XY_1 = num_XY - num_XY_2
            assert ((num_XY_2 + num_XY_1 )== num_XY)
        else:
            num_XY_1 = num_XY
        

        print(f'Processing {filename}')
        print(f'\tTime Series: {num_time_series}, Number of XY: {num_XY}, Number of channels: {num_channels}, Image size: {h} x {w}')     

        with alive_bar(num_time_series * num_XY) as bar:
            for i in range(num_time_series):
                XY_m_cherry_levels = []
                XY_m_cherry_RPUs = []
                XY_yfp_levels = []
                XY_yfp_RPUs = []

                XY_m_cherry_levels_2 = []
                XY_m_cherry_RPUs_2 = []
                XY_yfp_levels_2 = []
                XY_yfp_RPUs_2 = []

                for j in range(num_XY):
                    m_cherry = my_array.get_frame_2D(t = i, v = j, c = 1)
                    yfp = my_array.get_frame_2D(t = i, v = j, c = 2)
          
                    if m_cherry.mean() == 0.0 or yfp.mean() == 0.0:
                        for _ in range(num_XY):
                            bar()
                        break 
                    
                    left, right, top, bottom = get_edges_coord(my_array.get_frame_2D(t = i, v = j, c = 0))

                    if j < (XY2_start - 1):
                        #########################
                        m_cherry = m_cherry[top:bottom, left:right].mean()
                        yfp = yfp[top:bottom, left:right].mean()
                        #########################
                        XY_yfp_levels.append(yfp)
                        XY_yfp_RPUs.append((yfp - FL_BLANK) / (YFP_RPU - FL_BLANK))
                        XY_m_cherry_levels.append(m_cherry)
                        XY_m_cherry_RPUs.append((m_cherry - FL_CHERRY_BLANK) / (CHERRY_RPU - FL_CHERRY_BLANK))
                        
                    else:
                        #########################
                        m_cherry = m_cherry[top:bottom, left:right].mean()
                        yfp = yfp[top:bottom, left:right].mean()
                        #########################
                        XY_yfp_levels_2.append(yfp)
                        XY_yfp_RPUs_2.append((yfp - FL_BLANK) / (YFP_RPU - FL_BLANK))
                        XY_m_cherry_levels_2.append(m_cherry)
                        XY_m_cherry_RPUs_2.append((m_cherry - FL_CHERRY_BLANK) / (CHERRY_RPU - FL_CHERRY_BLANK))

                    bar()
                    
                if len(XY_m_cherry_levels) != 0 or len(XY_yfp_levels) != 0:
                    timeSeries.append(count)
                    
                    if np.isnan(XY_m_cherry_RPUs).any():
                        XY_m_cherry_RPUs = np.array(XY_m_cherry_RPUs)
                        XY_yfp_RPUs = np.array(XY_yfp_RPUs)
                        XY_m_cherry_RPUs = XY_m_cherry_RPUs[np.isfinite(XY_m_cherry_RPUs)]
                        XY_yfp_RPUs = XY_yfp_RPUs[np.isfinite(XY_yfp_RPUs)]
                        XY_m_cherry_RPUs_2 = np.array(XY_m_cherry_RPUs_2)
                        XY_yfp_RPUs_2 = np.array(XY_yfp_RPUs_2)
                        XY_m_cherry_RPUs_2 = XY_m_cherry_RPUs_2[np.isfinite(XY_m_cherry_RPUs_2)]
                        XY_yfp_RPUs_2 = XY_yfp_RPUs_2[np.isfinite(XY_yfp_RPUs_2)]

                    if np.isnan(XY_m_cherry_levels).any():
                        XY_m_cherry_levels = np.array(XY_m_cherry_levels)
                        XY_yfp_levels = np.array(XY_yfp_levels)
                        XY_m_cherry_levels = XY_m_cherry_levels[np.isfinite(XY_m_cherry_levels)]
                        XY_yfp_levels = XY_yfp_levels[np.isfinite(XY_yfp_levels)]
                        XY_m_cherry_levels_2 = np.array(XY_m_cherry_levels_2)
                        XY_yfp_levels_2 = np.array(XY_yfp_levels_2)
                        XY_m_cherry_levels_2 = XY_m_cherry_levels_2[np.isfinite(XY_m_cherry_levels_2)]
                        XY_yfp_levels_2 = XY_yfp_levels_2[np.isfinite(XY_yfp_levels_2)]

                    m_cherry_error.append((np.std(XY_m_cherry_RPUs))/(math.sqrt(num_XY_1)))
                    m_cherry_levels.append(np.array(XY_m_cherry_levels).mean())
                    m_cherry_RPUs.append(np.array(XY_m_cherry_RPUs).mean())

                    yfp_error.append((np.std(XY_yfp_RPUs))/(math.sqrt(num_XY_1)))
                    yfp_levels.append(np.array(XY_yfp_levels).mean())
                    yfp_RPUs.append(np.asarray(XY_yfp_RPUs).mean())

                    if (mult_experiment):
                        m_cherry_error_2.append((np.std(XY_m_cherry_RPUs_2))/(math.sqrt(num_XY_2)))
                        m_cherry_levels_2.append(np.array(XY_m_cherry_levels_2).mean())
                        m_cherry_RPUs_2.append(np.array(XY_m_cherry_RPUs_2).mean())

                        yfp_error_2.append((np.std(XY_yfp_RPUs_2))/(math.sqrt(num_XY_2)))
                        yfp_levels_2.append(np.array(XY_yfp_levels_2).mean())
                        yfp_RPUs_2.append(np.asarray(XY_yfp_RPUs_2).mean())


                        # first experiment
                        plot_activity(m_cherry_RPUs, timeSeries, m_cherry_error,
                            'm-Cherry_1', 
                            title,
                            dir_path, 
                            f'm-cherry_results_1\m-Cherry_{count}.png', 
                            'Average Total m-Cherry Intensity (RPU)', 
                            'red', 
                            'm-Cherry'
                            )
                        
                        plot_activity(yfp_RPUs, timeSeries, yfp_error,
                            'YFP_1', 
                            title,
                            dir_path, 
                            f'YFP_results_1\YFP_{count}.png', 
                            'Average Total YFP Intensity (RPU)', 
                            'green', 
                            'YFP'
                        )
                        # second experiment
                        plot_activity(m_cherry_RPUs_2, timeSeries, m_cherry_error_2,
                            'm-Cherry_2',
                            title_2, 
                            dir_path, 
                            f'm-cherry_results_2\m-Cherry_{count}.png', 
                            'Average Total m-Cherry Intensity (RPU)', 
                            'red', 
                            'm-Cherry'
                            )
                        
                        plot_activity(yfp_RPUs_2, timeSeries, yfp_error_2,
                            'YFP_2', 
                            title_2,
                            dir_path, 
                            f'YFP_results_2\YFP_{count}.png', 
                            'Average Total YFP Intensity (RPU)', 
                            'green', 
                            'YFP'
                        )
                    else:
                        plot_activity(m_cherry_RPUs, timeSeries, m_cherry_error,
                            'm-Cherry', 
                            title,
                            dir_path, 
                            f'm-cherry_results\m-Cherry_{count}.png', 
                            'Average Total m-Cherry Intensity (RPU)', 
                            'red', 
                            'm-Cherry'
                            )
                        
                        plot_activity(yfp_RPUs, timeSeries, yfp_error,
                            'YFP', 
                            title,
                            dir_path, 
                            f'YFP_results\YFP_{count}.png', 
                            'Average Total YFP Intensity (RPU)', 
                            'green', 
                            'YFP'
                        )

                count += 5

            
            # calc the time between nd2 files
            if len(time_gap) > 0:
                count -= 5
                count += time_gap.pop()
            
            my_array.close

         

    print('Making GIFS')
    # make gifs
    if mult_experiment:
        make_gif(os.path.join(dir_path, 'm-cherry_results_1'))
        make_gif(os.path.join(dir_path, 'YFP_results_1'))
        make_gif(os.path.join(dir_path, 'm-cherry_results_2'))
        make_gif(os.path.join(dir_path, 'YFP_results_2'))
    else:
        make_gif(os.path.join(dir_path, 'm-cherry_results'))
        make_gif(os.path.join(dir_path, 'YFP_results'))

    print('Making CSV')
    # make csv file 
    # make data directories
    if not os.path.exists(os.path.join(dir_path, f'data')):
            os.makedirs(os.path.join(dir_path, f'data'))
    
    data_path = os.path.join(dir_path, f'data')

    if mult_experiment:
        make_csv(data_path, 'm_cherry_1.csv', m_cherry_levels, m_cherry_RPUs, m_cherry_error, timeSeries)
        make_csv(data_path, 'm_cherry_2.csv', m_cherry_levels_2, m_cherry_RPUs_2, m_cherry_error_2, timeSeries)
        make_csv(data_path, 'yfp_1.csv', yfp_levels, yfp_RPUs, yfp_error, timeSeries)
        make_csv(data_path, 'yfp_2.csv', yfp_levels_2, yfp_RPUs_2, yfp_error_2, timeSeries)
    else:
        make_csv(data_path, 'm_cherry.csv', m_cherry_levels, m_cherry_RPUs, m_cherry_error, timeSeries)
        make_csv(data_path, 'yfp.csv', yfp_levels, yfp_RPUs, yfp_error, timeSeries)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Given directory of ND2 files, outputs a graph of the mean activity level")
    parser.add_argument(
        "-i", 
        "--input_path",
        type=str,
        required=True,
        help="Path to the directory of nd2 files"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=False,
        help="Output path where the plots are saved",
    )
    args = parser.parse_args()
    main(args)