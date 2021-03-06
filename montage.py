#!/usr/bin/python

"""montage -- create montages from a sequence of images.

Usage:
    montage.py [-a A] [-b S] [-t T] -o OUTPUT INPUT...
    montage.py (-h | --help)

Options:
    -b S, --blur-sigma=S       Amount to blur the mask [default: 3]
    -t T, --threshold=T        Threshold for foreground detection [default: 16]
    -a A, --alpha=A            Alpha value to use for blending [default: 0.9]
    -o OUTPUT, --output=OUTPUT Name of the output file
    -h, --help                 Show this screen.

(C) 2013 Henry de Valence <hdevalence@hdevalence.ca>
Licenced under GPLv2+
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import threshold
from PIL import Image
from docopt import docopt

def split_channels(image):
    """
    Given an image of shape (x, y, 3), return a tuple
    of monochrome images, one for each channel.
    """
    return (image[..., 0], image[..., 1], image[..., 2])


def median(images):
    """
    Given a tuple of monochrome images, stack them with a median filter.
    """
    return np.median(np.dstack(images), axis=2).astype(np.uint8)


def create_mask(foreground, background, blur_sigma, thresh, opacity):
    """
    Given two RGB images, foreground and background,
    give a mask of the areas where foreground differs from the
    background by more than thresh.

    Apply blur_sigma amount of blurring, and set the opacity of
    the nonzero parts of the mask to opacity.
    """
    blurred_fg = gaussian_filter(foreground, blur_sigma).astype(int)
    blurred_bg = gaussian_filter(background, blur_sigma).astype(int)
    diff = np.sum(np.abs(fg - bg), axis=2)
    diff = threshold(diff, threshmin=thresh, newval=0)
    diff = threshold(diff, threshmax=thresh+1, newval=opacity)
    diff = gaussian_filter(diff, blur_sigma).astype(np.uint8)
    return diff


def create_background(images):
    """
    Uses a median filter to build a background image.
    """
    # Apply median per channel and combine
    return np.dstack(map(median, zip(*map(split_channels, images))))


def merge_images(background, foregrounds, masks):
    """
    Successively compose foreground images onto background image
    with alpha channels given by the masks param.
    """
    bg_i = Image.fromarray(background)
    for fg, m in zip(foregrounds, masks):
        fg_i = Image.fromarray(fg)
        m_i = Image.fromarray(m)
        bg_i = Image.composite(fg_i, bg_i, m_i)
    return bg_i


if __name__ == "__main__":
    arguments = docopt(__doc__)
    images = [np.asarray(Image.open(f), dtype=np.uint8)
              for f in arguments['INPUT']]
    bg = create_background(images)
    sigma = float(arguments['--blur-sigma'])
    thresh_level = int(arguments['--threshold'])
    alpha = int(255*float(arguments['--alpha']))
    masks = [create_mask(fg, bg, sigma, thresh_level, alpha)
             for fg in images]
    composite = merge_images(bg, images, masks)
    composite.save(arguments['--output'])
