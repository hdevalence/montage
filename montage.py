#!/usr/bin/python

# (C) 2013 Henry de Valence <hdevalence@hdevalence.ca>
# Licenced under GPLv2+

import numpy as np
import scipy.ndimage
import scipy.stats
from PIL import Image

BLUR_SIGMA = 3


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


def blur(image):
    """
    Blur the image slightly.

    We use this to avoid noise splotches, when we mask the images.
    """
    return scipy.ndimage.filters.gaussian_filter(image, BLUR_SIGMA)


def mask(foreground, background, thresh):
    """
    Given two RGB images, foreground and background,
    give a mask of the areas where foreground differs from the
    background by more than thresh.
    """
    diff = np.abs(foreground.astype(int) - background.astype(int))
    m = np.sum(diff, axis=2)
    m = scipy.stats.threshold(m, threshmin=thresh, newval=0)
    m = scipy.stats.threshold(m, threshmax=thresh+1, newval=200)
    m = scipy.ndimage.filters.gaussian_filter(m, BLUR_SIGMA)
    return m.astype(np.uint8)


def create_background(images):
    """
    Uses a median filter to build a background image.
    """
    # Apply median per channel and combine
    return np.dstack(map(median, zip(*map(split_channels, images))))


def merge_images(background, foregrounds, masks):
    bg_i = Image.fromarray(background)
    for fg, m in zip(foregrounds, masks):
        fg_i = Image.fromarray(fg)
        m_i = Image.fromarray(m)
        bg_i = Image.composite(fg_i, bg_i, m_i)
    return bg_i


if __name__ == "__main__":
    import sys
    openImage = lambda f: np.asarray(Image.open(f), dtype=np.uint8)
    images = list(map(openImage, sys.argv[1:]))
    bg = create_background(images)
    bg_image = Image.fromarray(bg)
    bg_image.save("/tmp/test_bg.png")
    masks = list(map(lambda fg: mask(blur(fg), blur(bg), 16), images))
    composite = merge_images(bg, images, masks)
    composite.save("/tmp/test_bg_mixed.png")
