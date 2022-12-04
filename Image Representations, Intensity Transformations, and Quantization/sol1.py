import numpy as np
from imageio import imread, imwrite
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

GRAYSCALE = 1
RGB = 2
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    im = imread(filename)

    if representation == GRAYSCALE:
        im = rgb2gray(im)  # already normalized
    elif representation == RGB:
        im = im.astype(np.float64) / 255

    return im


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """
    im = read_image(filename, representation)
    if representation == GRAYSCALE:
        plt.imshow(im, cmap=plt.cm.gray, vmax=1, vmin=0)
    elif representation == RGB:
        plt.imshow(im)


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """
    reds = np.inner(RGB_YIQ_TRANSFORMATION_MATRIX[0], imRGB)
    greens = np.inner(RGB_YIQ_TRANSFORMATION_MATRIX[1], imRGB)
    blues = np.inner(RGB_YIQ_TRANSFORMATION_MATRIX[2], imRGB)

    return np.dstack((reds, greens, blues))


def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space
    :param imYIQ: height X width X 3 np.float64 matrix in the [0,1] range for
        the Y channel and in the range of [-1,1] for the I,Q channels
    :return: the image in the RGB space
    """
    inv = np.linalg.inv(RGB_YIQ_TRANSFORMATION_MATRIX)  # use inverse of transformation matrix
    reds = np.inner(inv[0], imYIQ)
    greens = np.inner(inv[1], imYIQ)
    blues = np.inner(inv[2], imYIQ)

    return np.dstack((reds, greens, blues))


def histogram_equalize(im_orig):
    """
    Perform histogram equalization on the given image
    :param im_orig: Input float64 [0,1] image
    :return: [im_eq, hist_orig, hist_eq]
    """
    # TODO set 255 as constant - max gray level

    # check if rgb or grayscale
    yiq = None
    if len(im_orig.shape) == 2:
        im = im_orig
    else:
        yiq = rgb2yiq(im_orig)
        im = yiq[:, :, 0]

    # 0. convert to int
    im = (im * 255).round().astype(np.uint8)

    # 1. compute image histogram
    hist_orig, bounds = np.histogram(im, 256, [0, 255])

    # 2. cumpute cumolative histogram
    cum_hist = np.cumsum(hist_orig)

    # 3. normalize cumulative histogram
    # 4. multiply by max gray level (255)
    # 5. stretch the result linearly in range [0, 255]
    # 6. round the values and map them
    m = (cum_hist != 0).argmax(axis=0)  # first non-zero element index
    T = (((cum_hist - cum_hist[m]) / (cum_hist[255] - cum_hist[m])) * 255).round()
    T = np.array(T)
    im_eq = T[im].astype(np.float64)

    # calculate final hist and normalize equalized image
    hist_eq, bounds = np.histogram(im_eq, 256, [0, 255])
    im_eq /= 255

    # update back to rgb if needed
    if yiq is not None:
        new_yiq = np.dstack((im_eq, yiq[:, :, 1], yiq[:, :, 2]))
        im_eq = yiq2rgb(new_yiq)

    return [im_eq, hist_orig, hist_eq]


def calculate_current_error(Z, Q, k, H):
    """
    Calculate error by given formula from lecture.
    :param Z: Borders which divide the histogram
    :param Q: Values the pixels in each partition will be mapped to
    :param k: n_quants - number of intensities
    :param H: The histogram
    :return: Value of the error
    """
    sum = 0
    for i in range(k):
        start = (np.floor(Z[i]) + 1).astype(np.uint8)
        end = np.floor(Z[i + 1]).astype(np.uint8)
        sum += np.sum(np.array([((Q[i] - g) ** 2) * H[g] for g in range(start, end + 1)]))
    return sum


def z_q(Z, H):
    """
    Calculates Q according to given Z
    :param Z: Given Z to calculate Q according to
    :param H: The histogram
    :return: Q
    """
    Q = []
    for i in range(len(Z) - 1):
        start = (np.floor(Z[i]) + 1).astype(np.uint8)
        end = np.floor(Z[i + 1]).astype(np.uint8)
        top = np.array([g * H[g] for g in range(start, end + 1)])
        bottom = np.array([H[g] for g in range(start, end + 1)])
        Q.append(np.sum(top) / np.sum(bottom))
    return np.array(Q)


def q_z(Q, n_quant):
    """
    Calculate Z according to given Q
    :param Q: Given Q to calculate Z according to
    :param n_quant: number of intensities
    :return: Z
    """
    Z = np.zeros(n_quant + 1)
    for i in range(1, n_quant):
        Z[i] = (Q[i - 1] + Q[i]) / 2
    Z[-1] = 255
    Z[0] = -1
    return np.array(Z)


def find_z_partition(im, H, n_quant):
    """
    Calculate initial partition Z such that each partition will have approx the same amount of pixels
    :param im: image
    :param H: histogram
    :param n_quant: number of intensities
    :return: Z - partition
    """
    partition_size = im.shape[0] * im.shape[1] // n_quant  # approx amount of pixels each partition
    Z = [-1]  # handle edge case 0
    i = 0

    for j in range(n_quant):
        cumulative_pixels = 0  # counter for total pixels in partition j (Z[j]:Z[j+1])

        while H[i] + cumulative_pixels < partition_size:
            cumulative_pixels += H[i]
            i += 1
        Z.append(i)

    Z[-1] = 255  # last partition should end in 255
    return Z

def quantize(im_orig, n_quant, n_iter):
    """
    Calculate quantize only on intensities by the algorithm given in lectures. Steps documented in code
    :param im_orig: original image
    :param n_quant: number of intensities
    :param n_iter: maximum amount of iterations to minimize error
    :return: quanitized image, array of errors
    """
    errors = []
    Q = []

    # 0. check if rgb or grayscale
    yiq = None  # will store yiq to save recalculation at the end
    im = im_orig
    if len(im_orig.shape) == 3:
        yiq = rgb2yiq(im_orig)
        im = yiq[:, :, 0]

    im = (im * 255).round().astype(np.uint8)  # set to 0-255 for histogram

    # 1. compute histogram
    H = np.histogram(im, 256, [0, 255])[0]

    # 2. initial Z
    Z = find_z_partition(im, H, n_quant)

    # 3. iterations
    for i in range(n_iter):
        # calculate Q and then Z according to it
        Q = z_q(Z, H)
        new_Z = q_z(Q, n_quant)

        errors.append(calculate_current_error(new_Z, Q, n_quant, H))  # calculate and append new error
        if np.array_equal(new_Z, Z):  # if converges (convergence defined in lectures as Z not changing)
            break
        Z = new_Z

    # map pixels to intensities
    new_im = np.zeros(256)
    for i in range(len(Q)):
        start = (np.floor(Z[i]) + 1).astype(np.uint8)
        end = np.floor(Z[i + 1]).astype(np.uint8)
        new_im[start: end + 1] = Q[i].round().astype(np.uint8)
    new_im[-1] = Q[-1].round().astype(np.uint8)

    new_im = new_im[im.astype(np.uint8)]

    new_im /= 255  # range 0-1

    # if rgb, convert back
    if yiq is not None:
        new_yiq = np.dstack((new_im, yiq[:, :, 1], yiq[:, :, 2]))
        new_im = yiq2rgb(new_yiq)

    return new_im, errors

def quantize_rgb(im_orig, n_quant):  # Bonus - optional
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input RGB image of type float64 in the range [0,1]
    :param n_quant: Number of intensities im_quant image will have
    :return:  im_quant - the quantized output image
    """

    """
    algorithm - Median Cut:
        1. Move all pixels into a single large bucket.
        2. Find the color channel (red, green, or blue) in the image with the greatest range.
        3. Sort the pixels by that channel values.
        4. Find the median and cut the region by that pixel.
        5. Repeat the process for both buckets until you have the desired number of colors.
    """
    pass



