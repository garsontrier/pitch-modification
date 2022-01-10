from psola import pitch_marking, divide_into_segments, change_pitch, psola
import librosa
from yin_algorithm import yin_pitchtracker, median_filter, pitch_to_samples
from scipy.io.wavfile import write
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def print_segments_pitch(pitch, segment_frames):
    pitch = np.asarray(pitch)
    pitch_list = []
    for key, value in segment_frames.items():
        pitch_list.clear()
        for i in value:
            pitch_list.append(pitch[i])
        length = len(pitch_list)
        pitch_in_seg = st.mean(pitch_list)
        print('Segment ' + str(key) + ' with frame length of ' + str(length) + ' and pitch: %.2f Hz' % pitch_in_seg)


def get_segment_pitch(pitch, segment_frames, seg_no):
    frames = segment_frames[seg_no]
    temp_pitch = []
    for i in frames:
        temp_pitch.append(pitch[i])
    seg_pitch = st.mean(temp_pitch)
    return seg_pitch

def plot_pitch_contours(pitch, new_pitch, wf):
    plt.plot(pitch, 'x', new_pitch, 'rx')
    plt.legend(['Original Pitch', 'Modified Pitch'])
    plt.ylabel('Hz')
    plt.xlabel('Frame No')
    plt.savefig(wf + '_contour.png', dpi=100)
    plt.show()


def plot_pitch_contour(pitch):
    plt.plot(pitch, 'x')
    plt.ylabel('Hz')
    plt.xlabel('Frame No')
    plt.show()


def plot_pitch_markings(frame_nos, y, marks):
    mark_list = []
    for frame in frame_nos:
        mark_list.extend(marks[frame])
        if len(mark_list) > 4:
            break
    bound = [mark_list[0]-20, mark_list[-1]+20]
    y = np.asarray(y)
    z = np.zeros_like(y)
    z[mark_list] = y[mark_list]
    plt.plot(y[bound[0]:bound[1]])
    plt.plot(z[bound[0]:bound[1]], 'x')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.savefig('markings.png', dpi=100)
    plt.show()

def mse(segment_frames, selected_segments, desired_pitch, c_pitch):
    d_pitch = c_pitch.copy()
    d_pitch = np.asarray(d_pitch)
    for i in range(len(selected_segments)):
        frames = segment_frames[selected_segments[i]]
        d_pitch[frames] = desired_pitch[i]
    c_pitch = np.asarray(c_pitch.copy())
    mse = (np.square(d_pitch - c_pitch)).mean(axis=0)
    return mse


if __name__ == '__main__':
    base_dir = "./"
    wf = "S020_2012572582"
    fs = 16000
    y, fs = librosa.load(base_dir + wf + '.wav', sr=fs)
    frame_size = 0.02
    frame_step = 0.02
    frame_len = int(frame_size * fs)

    pitches = yin_pitchtracker(y, frame_size, frame_step, fs)
    smoothed_pitch = median_filter(pitches, 5)
    samples = pitch_to_samples(pitches, fs)
    segment_frames = divide_into_segments(samples)
    print_segments_pitch(smoothed_pitch, segment_frames)
    print('Reminder: You can modify pitch in the interval of [pitch/2, 2*pitch]')
    print('Select segments whose pitch you want to modify:')
    message = ''
    selected_segments = []
    desired_pitch = []
    while message not in ['exit', 'Exit', 'EXIT']:
        print('Enter the number of the segment you want to work on:')
        seg_no = int(input())
        selected_segments.append(seg_no)
        seg_pitch = get_segment_pitch(smoothed_pitch, segment_frames, seg_no)
        print('Pitch of the segment: %.2f Hz' % seg_pitch)
        print('Enter the pitch value you want to set')
        desired_pitch.append(int(input()))
        message = input('Press Enter to add more segments or type exit to finish')


    y1, marks = psola(y, samples, selected_segments, desired_pitch, frame_len, fs)
    plot_pitch_markings(segment_frames[0], y, marks)
    new_pitches = yin_pitchtracker(y1, frame_size, frame_step, fs)
    ms_error = mse(segment_frames, selected_segments, desired_pitch, new_pitches)
    sm_new_pitch = median_filter(new_pitches, 5)
    write('./modified_' + wf + '.wav', fs, y1)
    smoothed_pitch = list(smoothed_pitch)
    pitches = list(pitches)
    int_pit = [int(i) for i in pitches]
    int_pit_m = [int(i) for i in new_pitches]
    plot_pitch_contours(int_pit, int_pit_m, wf)
    plot_pitch_contour(int_pit)
    plot_pitch_contour(int_pit_m)