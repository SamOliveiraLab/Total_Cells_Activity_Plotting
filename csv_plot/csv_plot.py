import os
import re
import argparse
import cv2 as cv
import numpy as np
import imageio.v2 as imageio
from matplotlib import pyplot as plt

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

def main(args):
    dir_path = f"{args.input_path}"
    res = []

    for path in os.listdir(dir_path):
        # check if its a file and is a file type csv
        if path.endswith('.csv') and os.path.isfile(os.path.join(dir_path, path)):
            res.append(path)
    print(res)

    title = input('Title: ')

    if not os.path.exists(os.path.join(dir_path, f'm-cherry_results')):
            os.makedirs(os.path.join(dir_path, f'm-cherry_results'))
    if not os.path.exists(os.path.join(dir_path, f'YFP_results')):
            os.makedirs(os.path.join(dir_path, f'YFP_results'))

    cherry_data = np.transpose(np.loadtxt(open(os.path.join(dir_path,res[0]), "rb"), delimiter=",", skiprows=1))
    yfp_data = np.transpose(np.loadtxt(open(os.path.join(dir_path,res[1]), "rb"), delimiter=",", skiprows=1))
    
    cherry_times = cherry_data[0]
    cherry_rpus = cherry_data[2]
    cherry_errors = cherry_data[3]

    yfp_times = yfp_data[0]
    yfp_rpus = yfp_data[2]
    yfp_errors = yfp_data[3]


    time = []
    rpus = []
    errors = []

    for i in range(len(cherry_times)):
        time.append(cherry_times[i])
        rpus.append(cherry_rpus[i])
        errors.append(cherry_errors[i])
        
        plot_activity(rpus, time, errors,
                            'm-Cherry', 
                            title,
                            dir_path, 
                            f'm-cherry_results\m-Cherry_{int(cherry_times[i])}.png', 
                            'Average Total m-Cherry Intensity (RPU)', 
                            'red', 
                            'm-Cherry'
                            )


    time = []
    rpus = []
    errors = []

    for i in range(len(yfp_times)):
        time.append(yfp_times[i])
        rpus.append(yfp_rpus[i])
        errors.append(yfp_errors[i])
        
        plot_activity(rpus, time, errors,
                            'YFP', 
                            title,
                            dir_path, 
                            f'YFP_results\YFP_{int(yfp_times[i])}.png', 
                            'Average Total YFP Intensity (RPU)', 
                            'green', 
                            'YFP'
                            )


    make_gif(os.path.join(dir_path, 'm-cherry_results'))
    make_gif(os.path.join(dir_path, 'YFP_results'))
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Given directory of the csv files, outputs a graph of the mean activity level")
    parser.add_argument(
        "-i", 
        "--input_path",
        type=str,
        required=True,
        help="Path to the directory of csv file"
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