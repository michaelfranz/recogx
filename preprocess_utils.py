import librosa
import numpy as np
from tqdm import tqdm_notebook


def get_max_duration(filenames):
    max_duration = 0
    for path in tqdm_notebook(filenames):
        duration = librosa.core.audio.get_duration(filename=path)
        if duration > max_duration:
            max_duration = duration
    return max_duration


def pad_input(input, target_len):
    # CNN needs input to be same length. This function truncates input if too long; pads input with zeros if troo short
    num_zeros_needed = target_len - len(input)
    if num_zeros_needed > 0:
        num_zeros_front = np.random.randint(num_zeros_needed)
        num_zeros_back = num_zeros_needed - num_zeros_front
        return np.pad(input, (num_zeros_front, num_zeros_back), mode='constant')
    else:
        return input[0: target_len]


kpre_emphasis_coeff = 0.97


def __pre_emphasis__(input):
    first_amp = input[0]
    all_amps_without_first = input[1:]
    all_amps_without_last = input[:-1]
    emphasized_input = np.append(first_amp, all_amps_without_first - kpre_emphasis_coeff * all_amps_without_last)
    return emphasized_input


def pipeline(sample_rate, input):

    emphasized_input = __pre_emphasis__(input)

    # apply dft, mel filter banks, logging, dct and normalization
    lifted_mfcc = librosa.feature.mfcc(
        y=emphasized_input.astype(float),
        sr=sample_rate,
        n_mfcc=12,
        dct_type=2,
        norm='ortho',
        lifter=22,
        n_fft = int(sample_rate * 0.025),
        hop_length= int(sample_rate * 0.01),
        power=2,
        center=False,
        window='hann',
        n_mels=40
    )

    return lifted_mfcc
