import numpy as np
import librosa
from scipy.signal import find_peaks
import statistics as st

def remove_unvoiced(y, samples, frame_len):
    unvoiced_pos = []
    unvoiced_win = []
    voiced_win = []
    index = 0
    win_no = 0
    y1 = []
    for i in range(len(samples)):
        if int(samples[i]) == 0: # from index to index+frame_len is unvoiced
            unvoiced_pos.append(index)
            #y = np.append(y[0:index], y[index+frame_len:])
            unvoiced_win.append(win_no)
        else:
            y1.append(list(y[index:index+frame_len]))
            voiced_win.append(win_no)
        index = index + frame_len
        win_no += 1
    y1 = np.asarray(y1)
    return y1, unvoiced_pos, unvoiced_win, voiced_win


def pitch_marking(y, samples, frame_len, tol=3):
    # marks maximum points in the pitch period
    peak_pos = {}
    rem = 0
    for i in range(len(samples)):
        pos = i*frame_len
        if samples[i] != 0 and samples[i-1] == 0 and i != 0:
            rem = 0
        if samples[i] != 0:
            temp_marks, rem = find_peaks_in_frame(y[pos:pos+frame_len], samples[i], rem, tol)
            temp_marks = temp_marks + pos
            peak_pos[i] = list(temp_marks)
    return peak_pos


def find_peaks_in_frame(frame, pitch, rem, tol):
    marks = []  # offset holds last found peak position
    pitch = int(pitch)
    interval = int(0.4*pitch)
    frame_len = len(frame)
    if rem == 0:
        peak_pos = find_max_peak(frame, 0, int(1.5*pitch)+tol)
        offset = peak_pos
        marks.append(offset)
    else:
        peak_pos = find_max_peak(frame, 0, pitch - rem + tol)
        offset = peak_pos
        marks.append(offset)
    while offset + pitch + tol < frame_len:
        if offset + pitch + interval > frame_len:
            peak_pos = find_max_peak(frame, offset + pitch - interval, frame_len)
        else:
            peak_pos = find_max_peak(frame, offset + pitch - interval, offset + pitch + interval)
        if offset < 0:
            print('neg')
        offset = offset + pitch - interval + peak_pos
        marks.append(offset)
    remainder = frame_len - offset
    return np.asarray(marks), remainder


def find_max_peak(arr, start, end):
    empty = True
    while empty:
        peaks, _ = find_peaks(arr[start:end])
        try:
            max_ind = np.argmax(arr[start + peaks])
            empty = False
        except ValueError:
            empty = True
            start -= 2
            end += 2
            if start < 0:
                start = 0
            if end > len(arr):
                end = len(arr)
    return peaks[max_ind]


def divide_into_segments(samples):
    segment_frames = {}
    temp_pitch = []
    frame_nos = []
    segment_no = 0
    for i in range(len(samples)):
        if samples[i] != 0:
            if len(frame_nos) == 0:
                frame_nos.append(i)
                temp_pitch.append(samples[i])
            elif abs(mean - samples[i]) > 10: # end segment
                segment_frames[segment_no] = frame_nos.copy()
                frame_nos.clear()
                temp_pitch.clear()
                frame_nos.append(i)
                temp_pitch.append(samples[i])
                segment_no += 1
            else:
                frame_nos.append(i)
                temp_pitch.append(samples[i])
            mean = st.mean(temp_pitch)
            #median = st.median(temp_pitch)
        else:
            if len(frame_nos) != 0:
                segment_frames[segment_no] = frame_nos.copy()
                frame_nos.clear()
                temp_pitch.clear()
                segment_no += 1
    segment_frames = delete_short_segments(segment_frames)
    return segment_frames


def delete_short_segments(segment_frames):
    for key, value in segment_frames.copy().items():
        if len(value) < 3:
            del segment_frames[key]
    no = 0
    ordered_seg_frames = {}
    for value in segment_frames.values():
        ordered_seg_frames[no] = value
        no += 1
    return ordered_seg_frames


def extract_frames(y, frame_start, frame_end, frame_len):
    frame_list = []
    for i in range(frame_start,frame_end):
        frame_list.append(list(y[i*frame_len:i*frame_len+frame_len]))
    return frame_list


def segment_windowing(y, segment, samples, marks):
    avg_pitch = 0
    no_marks = 0
    pitches_in_segment = []
    for i in segment:
        avg_pitch += samples[i]
        pitches_in_segment.append(samples[i])
        no_marks += len(marks[i])
    avg_pitch = int(round(avg_pitch/len(segment)))  # used to define length of each window
    pitches = np.asarray(pitches_in_segment)
    med_pitch = int(np.median(pitches))
    diff = np.abs(avg_pitch - med_pitch)  # to give an idea about error
    frames = np.zeros([no_marks, 2*(avg_pitch)])
    win = np.hamming(2*(avg_pitch))
    mark_no = 0
    for i in range(len(segment)):
        frame_no = segment[i]
        for mark in marks[frame_no]:
            frames[mark_no] = y[mark-avg_pitch:mark+avg_pitch]*win
            mark_no += 1

    return frames, avg_pitch, diff


def get_frames(y, start, end, frame_len): # end frame is not included
    frames = []
    for i in range(start, end):
        frames.append(list(y[i*frame_len:(i+1)*frame_len]))
    return frames


def overlap_add_with_new_pitch(seg_frames, seg_pitch, desired_pitch, seg_len):
    temp_vec = np.zeros([2*seg_len, ])
    shift = desired_pitch
    frame_len = seg_frames.shape[1]
    for i in range(seg_frames.shape[0]):
        temp_vec[i*shift:i*shift+frame_len] = temp_vec[i*shift:i*shift+frame_len] + seg_frames[i]
    last_pos = i * shift + frame_len

    if last_pos < seg_len:
        diff = seg_len - last_pos
        temp_vec[last_pos:seg_len] = temp_vec[last_pos-diff:last_pos]
    '''
    last_frame = i-2
    i += 1
    while last_pos < seg_len:
        temp_vec[i*shift:i*shift + frame_len] = temp_vec[i*shift:i*shift + frame_len] + seg_frames[last_frame]
        last_pos = i * shift + frame_len
        i += 1
    '''
    seg_vec = temp_vec[0:seg_len]
    return seg_vec


def change_segment_pitch(y, samples,  selected_segment, pitch_marks, desired_pitch, frame_len): # desired pitch in samples
    seg_frames, seg_pitch, diff = segment_windowing(y, selected_segment, samples, pitch_marks)
    segment_len = len(selected_segment)*frame_len # segment length must be preserved
    new_seg_vec = overlap_add_with_new_pitch(seg_frames, seg_pitch, desired_pitch, segment_len)
    return new_seg_vec


def change_pitch(y, samples, segment_frames, selected_segments, desired_pitch, frame_len, pitch_marks):

    for i in range(len(selected_segments)):
        frames_in_seg = segment_frames[selected_segments[i]]
        new_segment = change_segment_pitch(y, samples, segment_frames[selected_segments[i]], pitch_marks, desired_pitch[i], frame_len)
        seg_start = frames_in_seg[0]
        seg_end = frames_in_seg[-1]
        y[seg_start*frame_len:(seg_end+1)*frame_len] = new_segment
    return y

def psola(y, samples, selected_segments, desired_pitch, frame_len, fs):
    y1 = y.copy()
    desired_pitch = np.asarray(desired_pitch)
    desired_pitch = np.round(fs//desired_pitch)
    desired_pitch = list(desired_pitch)
    segment_frames = divide_into_segments(samples)
    pitch_marks = pitch_marking(y1, samples, frame_len, 3)
    y1 = change_pitch(y1, samples, segment_frames, selected_segments, desired_pitch, frame_len, pitch_marks)
    return y1, pitch_marks
