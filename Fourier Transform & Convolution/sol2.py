# WRITER : Ron Edri , ron.edri , 206933012
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from skimage.color import rgb2gray
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates

GRAYSCALE = 1
RGB = 2
AUDIO_RATE_CHANGE = "change_rate.wav"
AUDIO_SAMPLE_CHANGE = "change_samples.wav"
CONV = [[0.5, 0, -0.5]]


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


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

    # im /= 255
    return im


def DFT(signal):
    """
    Transform a 1D discrete signal to its Fourier representation
    :param signal: array of dtype float64 with shape (N,) or (N,1)
    :return: complex fourier signal
    """
    size = len(signal)
    x = np.arange(size)
    u = x.reshape((size, 1))

    F = signal.T.dot(np.exp(((-2j) * np.pi * u * x) / size))
    F = F.reshape(signal.shape)
    return F


def IDFT(fourier_signal):
    """
    Inverse Discrete Fourier Transform
    :param fourier_signal: array of dtype complex128 with shape (N,) or (N,1)
    :return: complex signal.
    """
    x = np.arange(len(fourier_signal))
    u = x.reshape((len(fourier_signal), 1))
    e = np.exp((2j * np.pi * x * u) / len(fourier_signal))
    idft = fourier_signal.T @ e
    return idft.reshape(fourier_signal.shape) / len(fourier_signal)


def DFT2(image):
    """
    Convert a 2D discrete signal to its Fourier representation.
    As seen in class, we can calculate 2 dim DFT with 1 dim DFT:
    Applying DFT for each row, then applying DFT for each column
    :param image: grayscale image of dtype float64, shape (M,N) or (M,N,1)
    :return: return shape should be the same as the shape of the input with real values
    """
    F = []
    for row in range(len(image)):
        current = image[row, :]
        F.append(DFT(current))

    F = np.array(F)
    for col in range(len(image[0])):
        current = F[:, col]
        F[:, col] = DFT(current)

    return F


def IDFT2(fourier_image):
    """
    Same implementation idea as DFT2 but with iDFT
    :param fourier_image: 2D array of dtype complex128, shape (M,N) or (M,N,1)
    :return: return shape should be the same as the shape of the input
    """
    F = []
    for row in range(len(fourier_image)):
        current = fourier_image[row, :]
        F.append(IDFT(current))

    F = np.array(F)
    for col in range(len(fourier_image[0])):
        current = F[:, col]
        F[:, col] = IDFT(current)

    return F


def change_rate(filename, ratio):
    """
    Changes the duration of an audio file by keeping the same samples, but changing the
    sample rate written in the file header.
    :param filename: string representing the path to a WAV file
    :param ratio: positive float64 representing the duration change, where 0.25 < ratio < 4
    :return:
    """
    samplerate, data = wavfile.read(filename)
    new_samplerate = int(samplerate * ratio)
    wavfile.write(AUDIO_RATE_CHANGE, new_samplerate, data)


def resize(data, ratio):
    """
    Resize data according to new ratio - followed algorithm seen in class where we drop N/2 highest frequencies
    due to being able to recreate them later (nyquist) when output is shorter, and pad when output is longer.
    Ratio of frequencies before DFT and after is 1:1, and high values are of least importance (?)
    :param data: 1D ndarray of dtype float64 or complex128(*) representing the original sample points
    :param ratio:
    :return: 1D ndarray of the dtype of data representing the new sample points
    """
    if ratio == 1:
        return data

    # 1. Compute fourier
    f = DFT(data)

    # 2. Bring u=0 to center
    f_centered = np.fft.fftshift(f)

    # 3. Crop from N to N/2 - remove N/2 high frequencies
    N = len(f)
    new_size = int(N / ratio)

    if ratio < 1:  # need to pad - audio slower thus longer
        left = (new_size - N) // 2
        right = left + 1
        f_res = np.pad(f_centered, (left, right))
    else:  # need tu cut high frequencies - audio faster thus shorter
        left = (N - new_size) // 2
        f_res = f_centered[left:left + new_size]  # clipping high frequencies

    # 4. Compute inverse fourier
    f_uncentered = np.fft.ifftshift(f_res)
    return IDFT(f_uncentered)


def change_samples(filename, ratio):
    """
    Changes the duration of an audio file by reducing the number of samples
    using Fourier
    :param filename: string representing the path to a WAV file
    :param ratio: positive float64 representing the duration change
    :return:
    """
    samplerate, data = wavfile.read(filename)
    new_data = resize(data, ratio)
    new_data = np.real(new_data).astype(data.dtype.type)  # remove complex numbers, set return type as input
    wavfile.write(AUDIO_SAMPLE_CHANGE, samplerate, new_data)


def resize_spectrogram(data, ratio):
    """
    Speeds up a WAV file, without changing the pitch, using spectrogram scaling. This
    is done by computing the spectrogram, changing the number of spectrogram columns,
    and creating back the audio
    :param data: 1D ndarray of dtype float64 representing the original sample points
    :param ratio: float64 representing the rate change of the WAV file
    :return: new sample points according to ratio with the same datatype as data, 0.25 < ratio < 4
    """
    # compute the spectogram
    spectogram = stft(data)

    # change the number of columns of spectogram
    resized = []
    for i in range(len(spectogram)):
        resized.append(resize(spectogram[i], ratio))
    resized = np.array(resized)

    # create back the audio
    return istft(resized).astype(data.dtype.type)


def resize_vocoder(data, ratio):
    """
    Speedups a WAV file by phase vocoding its spectrogram.
    Scaling the spectrogram as done before, but includes the correction of
    the phases of each frequency according to the shift of each window
    :param data: 1D ndarray of dtype float64 representing the original sample points
    :param ratio: positive float64 representing the rate change of the WAV file
    :return: given data rescaled according to ratio with the same datatype as data, 0.25 < ratio < 4
    """
    # compute the spectogram
    spectogram = stft(data)

    # change the number of columns of spectogram
    resized = phase_vocoder(spectogram, ratio)

    # create back the audio
    return istft(resized).astype(data.dtype.type)


def calculate_magnitude(dx, dy):  # function given in ex2 file
    return np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)


def conv_der(im):
    """
    Computes the magnitude of image derivatives
    :param im: grayscale images of type float64
    :return: magnitude of the derivative, with the same dtype and shape
    """
    mat = np.array(CONV)
    dx = signal.convolve2d(im, mat, mode='same')
    dy = signal.convolve2d(im, np.transpose(mat), mode='same')
    return calculate_magnitude(dx, dy).astype(im.dtype.type)


def der_along_axis(im):
    """
    Calculates the derivative of fourier along axis. Calculation is done the same way for x as image,
    or for y as transposed im sent as an argument
    :param im: original image for x, transposed image for y
    :return: derivative of fourier along axis
    """
    size = len(im[0])
    range_axis = np.arange(-size / 2, size / 2)  # as seen in class, we can take [-n/2,n/2]
    x_y_cons = 2j * np.pi
    return (x_y_cons / size) * range_axis * im


def fourier_der(im):
    """
    Computes the magnitude of the image derivatives using Fourier transform
    :param im: float64 grayscale image
    :return: magnitude of the derivative
    """
    # DFT + center so 0 will not be in middle
    dft = DFT2(im)
    centered = np.fft.fftshift(dft)

    # derivative of x,y directions
    der_x_fourier = der_along_axis(centered)
    der_y_fourier = der_along_axis(np.transpose(centered))

    # uncenter + IDFT
    uncentered_x = np.fft.ifftshift(der_x_fourier)
    uncentered_y = np.fft.ifftshift(np.transpose(der_y_fourier))
    dx = IDFT2(uncentered_x)
    dy = IDFT2(uncentered_y)
    return calculate_magnitude(dx, dy).astype(im.dtype.type)  # return type ??
