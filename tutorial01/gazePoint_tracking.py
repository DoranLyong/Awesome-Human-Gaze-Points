""" (ref) http://zbigatron.com/mapping-camera-coordinates-to-a-2d-floor-plan/
"""
import sys 
import os 
import os.path as osp 
import csv 

import cv2 
import numpy as np 
from tqdm import tqdm 
from omegaconf import OmegaConf #(ref) https://majianglin2003.medium.com/python-omegaconf-a33be1b748ab
import matplotlib.pyplot as plt 
from matplotlib import image



def draw_point_tracking(gaze_points:list, imagefile=None):

    
    img = image.imread(imagefile)
    
    old = (0, 0)
    for x, y, _ in tqdm(gaze_points):  # (x, y, z) coordinate 
        x, y = map(int, [x, y]) # (ref) https://wikidocs.net/64

        new = (x, y)

        plt.plot(old, new)
        old = (x, y)
    
    plt.show()






#%%
def main(cfg):

    intput_path = cfg.Required.inputPath # point source path 

    bg_img = cfg.Optional.bg_image   # background image path 



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


        draw_point_tracking(gaze_data, imagefile=bg_img ) 





#%% 
if __name__ == '__main__':
    cfg = OmegaConf.load("cfg.yaml")

    main(cfg)