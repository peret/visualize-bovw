import matplotlib
matplotlib.use('Agg') # for PNG rendering without a window appearing
import pylab as pl
import os
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import pil_to_array
from matplotlib.gridspec import GridSpec
from shutil import rmtree
import Image, ImageEnhance

import util
from util.data import storage as ds
import math

from sklearn.externals import joblib
from mpl_toolkits.axes_grid1 import make_axes_locatable

HIST_WIDTH = 1.5
HIST_ARRAY_WIDTH = 4# + 1
RADIUS = int(round(HIST_WIDTH * 1.4142135623730951 * (HIST_ARRAY_WIDTH + 1) * 0.5))
EXP_SCALE = -1.0/(HIST_ARRAY_WIDTH * HIST_ARRAY_WIDTH * 0.5);

# TODO: get rid of vmax?
# Unfortunately, joblib's delayed can't handle instance methods,
# so make this a function.
def visualize_image(self, importances, img_name, image_title, vmin, vmax):
    """Render the visualization for a single image
    and save it to a file.

    Args:
        self: The Visualization instance to use.
        importances: Array of importance values.
        img_name: Name of imagefile to visualize.
        image_title: Title that gets displayed above the visualization.
        vmin: Importances with an absolute value below vmin won't get
            rendered.
        vmax: Importances with an absolute value above vmax won't get
            rendered.
    """

    img_path = os.path.join(self.datamanager.PATHS["IMG"], img_name)
    if not os.path.isfile(img_path):
        return

    img = Image.open(os.path.join(self.datamanager.PATHS["IMG"], img_name))

    # convert image to grayscale
    if img.mode != "L":
        grayscale = img.convert('L').convert('RGB')
    else:
        img = img.convert('RGB')
        grayscale = img

    # reduce contrast of grayscale image
    contrast = ImageEnhance.Contrast(grayscale)
    grayscale = contrast.enhance(0.5)

    heatmap = self.heatmap_data(img_name, importances, (img.size[1], img.size[0]))
    has_negative = (heatmap < 0.0).any() # is this an image with negative weights?

    pos_heat = np.ma.masked_less(heatmap, vmin)
    if has_negative:
        pos_colormap = cm.Reds
        neg_heat = np.ma.masked_greater(heatmap, -vmin)
        neg_colormap = self.reverse_color_map(cm.Blues)
    else:
        pos_colormap = cm.Reds

    # create visualization
    gridspec = GridSpec(2, 2, height_ratios=[1,3])

    # original thumbnail
    img_subplot = pl.subplot(gridspec[0, 0])
    img_subplot.axis('off')
    img_subplot.imshow(img)

    # heatmap thumbnail
    heat_subplot = pl.subplot(gridspec[0, 1])
    heat_subplot.axis('off')
    axes_heat = heat_subplot.imshow(pos_heat, cmap=pos_colormap, alpha=1.0, vmin=vmin, vmax=vmax)
    ticks = np.linspace(vmin, vmax, 10)

    if has_negative:
        axes_neg = heat_subplot.imshow(neg_heat, cmap=neg_colormap, alpha=1.0, vmin=-vmax, vmax=-vmin)

    # combined subplot of grayscale image and heatmap
    combined_subplot = pl.subplot(gridspec[1, :])
    combined_subplot.axis('off')
    combined_subplot.imshow(grayscale)
    combined_subplot.imshow(pos_heat, cmap=pos_colormap, alpha=0.55)
    divider = make_axes_locatable(combined_subplot)
    cax = divider.append_axes("right", size=0.25, pad=0.33)
    pl.colorbar(axes_heat, cax=cax, ticks=ticks)

    if has_negative:
        combined_subplot.imshow(neg_heat, cmap=neg_colormap, alpha=0.55)
        cax = divider.append_axes("left", size=0.25, pad=0.33)
        neg_cb = pl.colorbar(axes_neg, cax=cax, ticks=-ticks)
        neg_cb.ax.yaxis.set_ticks_position("left")

    # save figure
    fig_path = os.path.join(self.datamanager.PATHS["RESULTS"], "_".join(image_title.lower().split()[0:2]))
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    pl.suptitle(image_title)
    pl.savefig(os.path.join(fig_path, img_name + ".png"), format="png")
    pl.clf()

class Visualization:
    """Contains methods to create the visualization for all
    images of a dataset.
    """
    def __init__(self, datamanager):
        self.datamanager = datamanager

    def absmax(self, a, b):
        """Returns the value with the higher absolute value."""
        return a if abs(a) > abs(b) else b

    def reverse_color_map(self, cmap):
        """Takes a matplotlib color map and returns
        a new color map in which the order of colors
        is reversed.
        """
        result = {}
        segments = cmap._segmentdata
        for c in ["red", "green", "blue"]:
            result[c] = tuple([(1.0 - x, y0, y1) for x, y0, y1 in segments[c][::-1]])
        return LinearSegmentedColormap('inverse_' + cmap.name, result)

    def get_max_importance(self, importances):
        return max(np.abs(importances))

    def get_min_importance(self, importances):
        """Return the third quartile as a lower bound for the importances
        to visualize.

        This is intended to reduce clutter in the images.
        """
        return np.percentile(np.abs(importances), 75)

    def heatmap_data(self, img_name, importances, img_size):
        """Creates a 2-dimensional float array, representing
        the heatmap of importances for the given image.

        The code to calculate a keypoint's environment is based on
        OpenCV's SIFT-implementation.
        Pixels that are contained in the environments of two or
        more distinct keypoints get assigned the maximum of
        the possible importances.

        Args:
            img_name: Filename of the image in question.
            importances: Array of feature importances.
            img_size: Tuple of (width, height) for the image.
        """
        keypoints_file = util.keypoints_name(img_name)
        indices_file = util.cluster_name(img_name)
        keypoints = ds.keypoints_from_file(os.path.join(self.datamanager.PATHS["KEYPOINTS"], keypoints_file))
        indices = [index[0] for index in ds.load_matrix(os.path.join(self.datamanager.PATHS["BOW"], indices_file))]

        assert len(keypoints) == len(indices), "Should be %d, but is %d" % (len(keypoints), len(indices))

        heatmap = np.zeros(img_size[0:2])
        rows = heatmap.shape[0]
        cols = heatmap.shape[1]

        for kp, index in zip(keypoints, indices):
            cosine = math.cos(math.radians(kp.angle)) / HIST_WIDTH
            sine = math.sin(math.radians(kp.angle)) / HIST_WIDTH

            for i in range(-RADIUS, RADIUS + 1):
                for j in range(-RADIUS, RADIUS + 1):
                    r = kp.y + i
                    c = kp.x + j

                    c_rot = j * cosine - i * sine
                    r_rot = j * sine + i * cosine;
                    rbin = r_rot + HIST_ARRAY_WIDTH/2 - 0.5
                    cbin = c_rot + HIST_ARRAY_WIDTH/2 - 0.5

                    if (-1 < rbin < HIST_ARRAY_WIDTH) and (-1 < cbin < HIST_ARRAY_WIDTH) and (0 <= r < rows) and (0 <= c < cols):
                        heatmap[r, c] = self.absmax(heatmap[r, c], importances[index])
        return heatmap

    def visualize_images(self, img_names, importances, image_titles):
        """Create visualizations for all images in the list img_names."""
        max_importance = self.get_max_importance(importances)
        min_importance = self.get_min_importance(importances)
        # Remove old results before visualizing all images,
        # to prevent mixing old and new visualizations.
        if os.path.isdir(self.datamanager.PATHS["RESULTS"]):
            rmtree(self.datamanager.PATHS["RESULTS"])
        joblib.Parallel(n_jobs=-1, pre_dispatch='2*n_jobs')(
            joblib.delayed(visualize_image)(self, importances, img_names[i], image_titles[i], min_importance, max_importance)
            for i in range(len(img_names)))
