import os
import argparse
import numpy as np
from matplotlib import pyplot as plt

def main(args):
    dir_path = f"{args.input_path}"
    res = []

    for path in os.listdir(dir_path):
        # check if its a file and is a file type csv
        if path.endswith('.csv') and os.path.isfile(os.path.join(dir_path, path)):
            res.append(path)
    print(res)

    title = input('Title: ')

    data = np.transpose(np.loadtxt(open(os.path.join(dir_path,res[0]), "rb"), delimiter=",", skiprows=1))
    
    time = data[0]
    rpus = data[2]
    errors = data[3]
    aTc = data[4]
    IPTG = data[5]
    M9_plain = data[6]

    time = np.around((np.divide(time, 60)), decimals=2)
    

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, gridspec_kw={'height_ratios': [10, 1, 1, 1]})
    fig.tight_layout()
    fig.set_figheight(7)
    ax1.plot(np.asarray(time), np.asarray(rpus), color='#f69b23')
    ax1.set_ylim(ymin=0, ymax=1.4) 

    ax1.fill_between(
                np.asarray(time), 
                np.asarray(rpus) - np.asarray(errors), 
                np.asarray(rpus) + np.asarray(errors),
                color="#ededed",
                alpha=0.5
            )
    ax1.set_ylabel('Average Total YFP Intensity (RPU)')
    ax1.set_xlabel('Time (h)')
    ax4.set_xlabel('Time (h)')
    ax1.legend(['YFP'])


    ax2.step(np.asarray(time), np.asarray(aTc), color = 'black')
    ax2.set_ylabel('aTC')
    ax3.step(np.asarray(time), np.asarray(IPTG), color = 'black')
    ax3.set_ylabel('IPTG')
    ax4.step(np.asarray(time), np.asarray(M9_plain), color = 'black')
    ax4.set_ylabel('M9_plain')


    axes = [ax1, ax2, ax3, ax4]
    for ax in axes:
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlim(xmin=0)

    # rest of the input 
    axes = [ax2, ax3, ax4]
    for ax in axes:
        ax.set_ylim(ymin= -0.25, ymax= 1.25)
        ax.label_outer()
        ax.set_yticks([])

    fig.text(0, 0.25, 'Input', rotation=90)

    fig.savefig(os.path.join(dir_path, f'{title}.svg'), format='svg', dpi = 1600, bbox_inches='tight')
    fig.savefig(os.path.join(dir_path, f'{title}.png'), format='png', dpi = 1600, bbox_inches='tight')


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