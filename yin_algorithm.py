import numpy as np


#this code is taken from: https://github.com/amackillop/pyYIN

# ### Yin pitch tracking algorithm

def divide_into_frames(y, frame_size, frame_stride, fs):
    frame_len = int(fs * frame_size)  # number of samples in a single frame
    frame_step = int(fs * frame_stride)  # number of overlapping samples
    total_frames = int(np.ceil(float(np.abs(len(y) - frame_len)) / frame_step))
    if frame_len * total_frames > len(y):
        padded_y = np.append(np.array(y), np.zeros(frame_len * total_frames - len(y)))
    else:
        padded_y = y[0:frame_len * total_frames]
    framed_y = np.zeros((total_frames, frame_len))
    for i in range(total_frames): # rectangular win
        framed_y[i] = padded_y[i * frame_step: i * frame_step + frame_len]

    return framed_y


def calculate_difference(signal):
    half_len_signal = len(signal) // 2
    tau = 0
    autocorr = np.zeros(half_len_signal)
    for tau in range(half_len_signal):
        for i in range(half_len_signal):
            diff = signal[i] - signal[i + tau]
            autocorr[tau] += diff ** 2

    return autocorr


def normalize_with_cumulative_mean(autocorr, halflen):
    new_autocorr = autocorr
    new_autocorr[0] = 1
    running_sum = 0.0
    for tau in range(1, halflen):
        running_sum += autocorr[tau]
        new_autocorr[tau] = autocorr[tau] / ((1 / tau) * running_sum)

    return new_autocorr


def absolute_threshold(new_autocorr, halflen, threshold):
    temp = np.array(np.where(new_autocorr < threshold))
    if (temp.shape == (1, 0)):
        tau = -1
    else:
        tau = temp[:, 0][0]
    return tau


def parabolic_interpolation(new_autocorr, tau, frame_len):
    if tau > 1 and tau < (frame_len // 2 - 1):
        alpha = new_autocorr[tau - 1]
        beta = new_autocorr[tau]
        gamma = new_autocorr[tau + 1]
        improv = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
    else:
        improv = 0

    new_tau = tau + improv
    return new_tau


def yin_pitchtracker(y, frame_size, frame_step, sr):
    framed_y = divide_into_frames(y, frame_size, frame_step, sr)
    frame_len = int(frame_size * sr)
    pitches = []
    for i in range(len(framed_y)):
        autocorr = calculate_difference(framed_y[i])
        new_autocorr = normalize_with_cumulative_mean(autocorr, frame_len // 2)
        tau = absolute_threshold(new_autocorr, frame_len // 2, 0.16)
        new_tau = parabolic_interpolation(new_autocorr, tau, frame_len)
        if (new_tau == -1):
            pitch = 0
        else:
            pitch = sr / new_tau

        if pitch >= 0:
            pitches.append(pitch)
    return pitches


# added by me
def median_filter(pitches, n):
    pitch = np.asarray(pitches)
    for i in range(len(pitch) - 2):
        window = pitch[i:i+n]
        pitch[i+2] = np.median(window)
    return pitch

def pitch_to_samples(pitches, fs):
    pitch = np.asarray(pitches)
    periods = 1/pitch
    samples = np.ceil(periods*fs)
    samples[samples == np.inf] = 0
    samples[samples < 0] = 0
    return samples

