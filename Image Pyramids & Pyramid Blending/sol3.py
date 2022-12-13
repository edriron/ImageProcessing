# WRITER : Ron Edri , ron.edri , 206933012
import numpy as np
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
from imageio import imread
from skimage.color import rgb2gray
import os

CONV_FILTER = [1, 1]


def convolve_with_filter(im, filter):
    convolved_im = convolve(im, filter)  # convolve over rows
    convolved_im = convolve(convolved_im, filter.T)  # convolve over cols
    return convolved_im


def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    # 1. convolve with blur_filter
    convolved_im = convolve_with_filter(im, blur_filter)

    # 2. sub-sample, select every second pixel in every second row
    sub_sampled_im = convolved_im[::2, ::2]
    return sub_sampled_im


def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    # 1. zero padding (a1, 0, a2, 0, ..)
    padded_im = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    padded_im[::2, ::2] = im

    # 2. blur
    convolved_im = convolve_with_filter(padded_im, 2 * blur_filter)
    return convolved_im


def get_gaussian_filter(filter_size):
    """
    returns gaussian filter of size
    :param size: an odd scalar that represents a squared filter
    :return: gaussian filter
    """
    c = np.array(CONV_FILTER)
    convolved = c
    for i in range(filter_size):
        convolved = np.convolve(convolved, c)

    return (convolved / np.sum(convolved)).reshape(1, -1)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    G = [im]
    filter_vec = get_gaussian_filter(filter_size - 2)
    for level in range(1, max_levels):
        reduced = reduce(G[-1], filter_vec)
        if reduced.shape[0] < 16 or reduced.shape[1] < 16:
            break
        G.append(reduced)

    return G, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    G, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    L = []
    for level in range(len(G) - 1):
        L.append(G[level] - expand(G[level + 1], filter_vec))

    L.append(G[-1])
    return L, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """
    # expand all L to the size of original pyramid
    expanded_l = [lpyr[0]]  # original image
    for i in range(1, len(lpyr)):
        im = lpyr[i]
        for j in range(i):  # pyramid[i] needs to be expanded i times
            im = expand(im, filter_vec)
        expanded_l.append(im)

    # multiply each level by corresponding weights
    for i in range(len(expanded_l)):
        expanded_l[i] *= coeff[i]

    # sum all up
    return sum(expanded_l)


def render_pyramid(pyr, levels):
    """
    Render the pyramids as one large image with 'levels' smaller images
        from the pyramid
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    :return: res a single black image in which the pyramid levels of the
            given pyramid pyr are stacked horizontally.
    """
    levels = min(levels, len(pyr))

    # stretch values to [0, 1]
    for i in range(levels):
        pyr[i] -= np.min(pyr[i])
        m = np.max(pyr[i])
        if m == 0:
            m = 1
        pyr[i] *= 255 / m

    # calculate hxw for rendered image
    width = len(pyr[0])
    height = 0
    for i in range(levels):
        height += len(pyr[i][0])

    # create black image with required shape
    res = np.zeros((width, height))

    # 'paste' first image at 0,0
    m, n = pyr[0].shape
    res[:m, :n] = pyr[0]

    # 'paste' all the rest
    prev_m, prev_n = 0, 0
    for i in range(1, levels):
        prev_m += pyr[i - 1].shape[0]
        prev_n += pyr[i - 1].shape[1]
        m, n = pyr[i].shape
        res[:m, prev_n: prev_n + n] = pyr[i]

    return res


def display_pyramid(pyr, levels):
    """
    display the rendered pyramid
    """
    im = render_pyramid(pyr, levels)
    plt.imshow(im, cmap=plt.cm.gray)
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    # 1. Construct Laplacian pyramids L1 and L2 for the input images im1 and im2, respectively.
    L1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]

    # 2. Construct a Gaussian pyramid Gm for the provided mask (convert it first to np.float64).
    Gm = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)[0]

    # 3. Construct the Laplacian pyramid Lout of the blended image for each level k
    size = len(Gm)
    L_out = [Gm[k] * L1[k] + (1 - Gm[k]) * L2[k] for k in range(size)]

    # 4. Reconstruct the resulting blended image from the Laplacian pyramid Lout (using ones for coefficients).
    coeff = [1 for _ in range(size)]
    im = laplacian_to_image(L_out, filter_vec, coeff)

    return im


def pyramid_blending_rgb(im1, im2, mask, max_levels, filter_size_im,
                         filter_size_mask):
    """
    blending on each color channel separately (on red, green and blue) and then combine the results into a single image
    :return:
    """
    r = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, max_levels, filter_size_im, filter_size_mask)
    g = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, max_levels, filter_size_im, filter_size_mask)
    b = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, max_levels, filter_size_im, filter_size_mask)
    im = np.dstack((r, g, b))
    im = np.where(im > 1, 1, im)
    im = np.where(im < 0, 0, im)
    return im


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    im = imread(filename)

    if representation == 1:
        im = rgb2gray(im)  # already normalized
    elif representation == 2:
        im = im.astype(np.float64) / 255
    return im


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def display_result(im1, im2, mask, out, representation):
    p = plt.figure(figsize=(4, 4))
    titles = ["Image 1", "Image 2", "Mask", "Blended"]
    for i, im in enumerate([im1, im2, mask, out]):
        p.add_subplot(221 + i)
        plt.tight_layout()
        plt.title(titles[i])
        if i == 2 or representation == 1:
            plt.imshow(im, cmap=plt.cm.gray)
        else:
            plt.imshow(im)
    plt.show()


def blending_example1():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im1 = read_image(relpath('externals/jet.jpg'), 2)
    im2 = read_image(relpath('externals/sky.jpg'), 2)
    mask = read_image(relpath('externals/sky_jet_mask.jpg'), 1).astype(np.bool)
    max_levels, filter_size_im, filter_size_mask = 8, 7, 13
    out = pyramid_blending_rgb(im1, im2, mask, max_levels, filter_size_im, filter_size_mask)
    display_result(im1, im2, mask, out, 2)
    return im1, im2, mask, out


def blending_example2():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im2 = read_image(relpath('externals/cup.jpg'), 2)
    im1 = read_image(relpath('externals/vortex.jpg'), 2)
    mask = read_image(relpath('externals/cup_vortex_mask.jpg'), 1).astype(np.bool)
    max_levels, filter_size_im, filter_size_mask = 4, 15, 21
    out = pyramid_blending_rgb(im1, im2, mask, max_levels, filter_size_im, filter_size_mask)
    display_result(im1, im2, mask, out, 2)
    return im1, im2, mask, out