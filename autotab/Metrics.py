"""Frame-level pitch and tablature precision / recall / F-measure."""

import numpy as np

from autotab.param import STRING_MIDI_PITCHES


def tab2pitch(tab):
    pitch_vector = np.zeros(44)
    for string_num in range(len(tab)):
        fret_class = np.argmax(tab[string_num], -1)
        if fret_class > 0:  # 0 means the string is not played
            pitch_vector[fret_class + STRING_MIDI_PITCHES[string_num] - 41] = 1
    return pitch_vector


def tab2bin(tab):
    tab_arr = np.zeros((6, 20))
    for string_num in range(len(tab)):
        fret_class = np.argmax(tab[string_num], -1)
        if fret_class > 0:
            tab_arr[string_num][fret_class - 1] = 1
    return tab_arr


def _safe_div(numerator, denominator):
    return float(numerator) / denominator if denominator else 0.0


def pitch_precision(pred, gt):
    pitch_pred = np.array(list(map(tab2pitch, pred)))
    pitch_gt = np.array(list(map(tab2pitch, gt)))
    return _safe_div(np.sum(pitch_pred * pitch_gt), np.sum(pitch_pred))


def pitch_recall(pred, gt):
    pitch_pred = np.array(list(map(tab2pitch, pred)))
    pitch_gt = np.array(list(map(tab2pitch, gt)))
    return _safe_div(np.sum(pitch_pred * pitch_gt), np.sum(pitch_gt))


def pitch_f_measure(pred, gt):
    p, r = pitch_precision(pred, gt), pitch_recall(pred, gt)
    return _safe_div(2 * p * r, p + r)


def tab_precision(pred, gt):
    tab_pred = np.array(list(map(tab2bin, pred)))
    tab_gt = np.array(list(map(tab2bin, gt)))
    return _safe_div(np.sum(tab_pred * tab_gt), np.sum(tab_pred))


def tab_recall(pred, gt):
    tab_pred = np.array(list(map(tab2bin, pred)))
    tab_gt = np.array(list(map(tab2bin, gt)))
    return _safe_div(np.sum(tab_pred * tab_gt), np.sum(tab_gt))


def tab_f_measure(pred, gt):
    p, r = tab_precision(pred, gt), tab_recall(pred, gt)
    return _safe_div(2 * p * r, p + r)


def tab_disamb(pred, gt):
    return _safe_div(tab_precision(pred, gt), pitch_precision(pred, gt))
