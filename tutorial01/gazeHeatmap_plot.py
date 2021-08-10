import sys 
import os 
import os.path as osp 
import csv 


import numpy as np 
from tqdm import tqdm 
from omegaconf import OmegaConf #(ref) https://majianglin2003.medium.com/python-omegaconf-a33be1b748ab
import matplotlib.pyplot as plt 
from matplotlib import image

from utils import ( draw_display, 
                    gaussian_kernel,
                    )
#%% 
def draw_heatmap(gazepoints, dispsize, imagefile=None, alpha=0.5, savefilename=None, gaussianwh=200, gaussiansd=None):
    """ Draws a heatmap of the provided fixations, optionally drawn over an image, 
        and optionally allocating more weight to fixations with a higher duration.
    
    [ arguments ]
    gazepoints		-	a list of gazepoint tuples (x, y)
    
    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    [ keyword arguments ]
    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    alpha		-	float between 0 and 1, indicating the transparancy of
                    the heatmap, where 0 is completely transparant and 1
                    is completely untransparant (default = 0.5)
    savefilename	-	full path to the file in which the heatmap should be
                    saved, or None to not save the file (default = None)
    returns
    fig			-	a matplotlib.pyplot Figure instance, containing the
                    heatmap
    """

    # =========================================== # 
    #           Show the background image         # 
    # =========================================== #
    fig, ax = draw_display(dispsize, imagefile=imagefile)
    


    # =========================================== # 
    #    Draw the Heatmap with Gaussian Matrix    # 
    # =========================================== #
    
    gwh = gaussianwh   # width and height of gaussian matrix
    gsdwh = gwh / 6 if (gaussiansd is None) else gaussiansd
    gaus = gaussian_kernel(gwh, gsdwh)
    
    # matrix of zeroes
    strt = gwh / 2

    heatmapsize = int(dispsize[1] + 2 * strt), int(dispsize[0] + 2 * strt)  # (height, width)
    heatmap = np.zeros(heatmapsize, dtype=float)
    
    
    # create heatmap
    for i in range(0, len(gazepoints)):
        # get x and y coordinates
        x = strt + gazepoints[i][0] - int(gwh / 2)
        y = strt + gazepoints[i][1] - int(gwh / 2)

        x, y = map(int, [x, y])

        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh]
            vadj = [0, gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x - dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y - dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * gazepoints[i][2]
            except:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            heatmap[y:y + gwh, x:x + gwh] += gaus * gazepoints[i][2]

    # resize heatmap
    strt = int(strt)

    heatmap = heatmap[strt:dispsize[1] + strt, strt:dispsize[0] + strt]
    # remove zeros
    lowbound = np.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = np.NaN
    # draw heatmap on top of image
    ax.imshow(heatmap, cmap='jet', alpha=alpha)

    # FINISH PLOT
    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)

    plt.show()

    return fig








#%% 
def main(cfg):
    
    intput_path = cfg.Required.inputPath 
    display_width = cfg.Required.displayWidth 
    display_height = cfg.Required.displayHeight 

    alpha = cfg.Optional.alpha 
    output_name = cfg.Optional.outputName 
    bg_img = cfg.Optional.bg_image 

    nGaussian = cfg.AdvancedOpt.sizeGM  # matrix size of Gaussian distirbution 
    sd = cfg.AdvancedOpt.stdGM  # 




    with open(intput_path) as f: 
        reader = csv.reader(f)
        raw = list(reader)  # gaze points 


        # =========================================== # 
        # Get the gaze point data in the integer type # 
        # =========================================== #
        gaze_data = [] 

        if len(raw[0]) == 2: 
            gaze_data = list( map(lambda q: (int(q[0]), int(q[1]), 1), raw) )   # cast the point element into the integer
                                                                                # (ref) https://wikidocs.net/64
        
        else: 
            gaze_data =  list(map(lambda q: (int(q[0]), int(q[1]), int(q[2])), raw)) # 3D gaze point? 


        # =========================================== # 
        #           Draw the gaze point heatmap       # 
        # =========================================== #
        draw_heatmap(gaze_data, (display_width, display_height), alpha=alpha, savefilename=output_name, imagefile=bg_img, gaussianwh=nGaussian, gaussiansd=sd)




#%% 
if __name__ == "__main__":
    cfg = OmegaConf.load("cfg.yaml")

    main(cfg)