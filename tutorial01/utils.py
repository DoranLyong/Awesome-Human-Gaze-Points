import os 

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import image


#%%
def draw_display(dispsize, imagefile=None):
    """Returns a matplotlib.pyplot Figure and its axes, with a size of dispsize, 
        a black background colour, and optionally with an image drawn onto it. 
    
    [ arguments ] 
    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)
    
    [ keyword arguments ]
    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)

    [ returns ]
    fig, ax		-	matplotlib.pyplot Figure and its axes: field of zeros
                    with a size of dispsize, and an image drawn onto it
                    if an imagefile was passed
    """

    # construct screen (black background)
    
    screen = np.zeros((dispsize[1], dispsize[0], 3), dtype='float32')


    # =========================================== # 
    #           Load the background image         # 
    # =========================================== #    
    if imagefile != None:
        # check if the path to the image exists
        if not os.path.isfile(imagefile): # (ref) https://docs.python.org/ko/3/library/os.path.html#os.path.isfile
            raise Exception(f"ERROR in draw_display: imagefile not found at '{imagefile}'" )

        img = image.imread(imagefile)
        h, w = img.shape[:2]


        """ 입력되는 이미지의 사이즈가 Gaze points의 범위보다 작아도 
            heatmap을 그릴 수 있도록 한다. 

            e.g., gaze points의 범위는 [:x, :y]=[:1024, :768] 이고
            입력되는 이미지의 shape은 (640, 480)이면, 포인트가 찍히는 범위가 더 넓기 때문에 
            zero-padding으로 입력 이미지의 사이즈를 늘려야 indexing error를 막을 수 있다. 

            (이 부분은 사실 테스트용 코드이기 때문에 존재한다. 실전에서는 없어질 부분)
        """
        # x and y position of the image on the display
        # if an image location has been passed, draw the image
        x = int(dispsize[0] / 2 - w / 2)
        y = int(dispsize[1] / 2 - h / 2)

        # draw the image on the screen
        screen[y:y + h, x:x + w, :] += img


    # =========================================== # 
    #         Display the background image        # 
    # =========================================== #   
    dpi = 100.0 # dots per inch
                # (ref) https://www.delftstack.com/ko/howto/matplotlib/how-to-plot-and-save-a-graph-in-high-resolution/
                # (ref) https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size/47639545



    # determine the figure size in inches
    figsize = (dispsize[0] / dpi, dispsize[1] / dpi)

    # create a figure
    fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])    # (ref) https://enjoyiot.tistory.com/entry/Visualization-with-Matplotlib-and-Pandas-2-Matplotlib-%ED%99%9C%EC%9A%A9%ED%95%98%EA%B8%B0
                                        # (ref) https://stackoverflow.com/questions/43326680/what-are-the-differences-between-add-axes-and-add-subplot
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.axis([0, dispsize[0], 0, dispsize[1]])
    ax.imshow(screen)   # (ref) https://stackoverflow.com/questions/45238306/why-does-imshows-extent-argument-flip-my-images
                        # (ref) https://matplotlib.org/stable/tutorials/intermediate/imshow_extent.html

#    plt.show()   
    return fig, ax




#%%
def gaussian_kernel(x, sx, y=None, sy=None):


    """Returns an array of np arrays (a matrix) containing values between
        1 and 0 in a 2D Gaussian distribution
    
    [ arguments ] 
    x		-- width in pixels
    sx		-- width standard deviation

    [ keyword argments ] 
    y		-- height in pixels (default = x)
    sy		-- height standard deviation (default = sx)
    """

    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sy == None:
        sy = sx

    # centers
    xo = x / 2
    yo = y / 2

    # matrix of zeros
    M = np.zeros([y, x], dtype=float)  # (ref) https://numpy.org/doc/stable/reference/generated/numpy.zeros.html

    # Gaussian matrix
    # (ref) https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy/29731818#29731818
    # (ref) https://gist.github.com/thomasaarholt/267ec4fff40ca9dff1106490ea3b7567
    from scipy import signal
    
    gkern1d = signal.gaussian(x, std=sx)
    M = np.outer(gkern1d, gkern1d)  # (ref) https://numpy.org/doc/stable/reference/generated/numpy.outer.html


    plt.imsave("./Outputs/Gaussian_kernel.png", M)  # (ref) https://stackoverflow.com/questions/30941350/how-are-images-actually-saved-with-skimage-python

    return M

