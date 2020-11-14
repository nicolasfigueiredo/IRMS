import librosa
import numpy as np
from irms import stft_zoom
from irms import detect_musical_regions
from irms.classes import SingleResSpectrogram, MultiResSpectrogram


def irms(y, k, kernel, model, pct, sr=44100, n_fft=512, hop_size=512, normalize=True):
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_size))
    time_span = [0, len(y)/sr]
    x_axis, y_axis = stft_zoom.get_axes_values(sr, 0, time_span, spec.shape)
    base_spec = SingleResSpectrogram(spec, x_axis, y_axis)
    multires_spec = MultiResSpectrogram(base_spec)
    indices, original_shape = detect_musical_regions.detect_musical_regions(model, spec, kernel=kernel, mode='pct', pct_or_threshold=pct, n_fft=n_fft, hop_size=hop_size)
    to_be_refined = detect_musical_regions.musical_regions_to_ranges(indices, original_shape, x_axis, y_axis, kernel, n_fft=n_fft, hop_size=hop_size)

    stft_zoom.set_signal_bank(y, kernel, n_fft=n_fft)

    for subregion in to_be_refined:
        freq_range = subregion[0]
        time_range = subregion[1]
        spec_zoom, x_axis, y_axis, new_sr, window_size, hop_size = stft_zoom.stft_zoom(y, freq_range, time_range, sr=sr, original_window_size=n_fft, k=k)
        refined_subspec = SingleResSpectrogram(spec_zoom, x_axis, y_axis)
        multires_spec.insert_zoom(multires_spec.base_spec, refined_subspec, freq_range, zoom_level=1, normalize=normalize)

    return multires_spec


def irms_multilevel(y, k_list, kernel_list, model, pct_list, sr=44100, n_fft=512, hop_size=512, normalize=True):
    o_n_fft = n_fft
    o_hop_size = hop_size
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_size))
    time_span = [0,len(y)/sr]
    x_axis, y_axis = stft_zoom.get_axes_values(sr, 0, time_span, spec.shape) 
    base_spec = SingleResSpectrogram(spec, x_axis, y_axis)
    multires_spec = MultiResSpectrogram(base_spec)

    kernel = kernel_list[0]
    indices, original_shape = detect_musical_regions.detect_musical_regions(model, spec, kernel=kernel, mode='pct', pct_or_threshold=pct_list[0], n_fft=n_fft, hop_size=hop_size)
    to_be_refined = detect_musical_regions.musical_regions_to_ranges(indices, original_shape, x_axis, y_axis, kernel, n_fft=n_fft, hop_size=hop_size)
    stft_zoom.set_signal_bank(y,kernel, n_fft=n_fft)

    for subregion in to_be_refined:
        freq_range = subregion[0]
        time_range = subregion[1]
        spec_zoom, x_axis, y_axis, new_sr, window_size, hop_size = stft_zoom.stft_zoom(y, freq_range, time_range, sr=sr, original_window_size=n_fft, k=k_list[0])
        refined_subspec = SingleResSpectrogram(spec_zoom, x_axis, y_axis, n_fft=window_size, hop_size=hop_size, sr=new_sr)
        multires_spec.insert_zoom(multires_spec.base_spec, refined_subspec, freq_range, zoom_level=1, normalize=normalize)

    i = 1
    for kernel in kernel_list[1:]:
        i += 1
#         print(k_list[i-1])
        if i == 2:
            spec_list = multires_spec.first_zoom
        elif i == 3:
            spec_list = multires_spec.second_zoom
        elif i == 4:
            spec_list = multires_spec.third_zoom

        to_be_further_refined = []
        for spec_zoom in spec_list:
            spec = np.array(spec_zoom.spec, dtype=float)
            x_axis = spec_zoom.x_axis
            y_axis = spec_zoom.y_axis
            sr = spec_zoom.sr
            window_size = spec_zoom.n_fft
            hop_size = spec_zoom.hop_size

            if len(y_axis) == 1 or len(x_axis) == 1:
                continue

            indices, original_shape = detect_musical_regions.detect_musical_regions(model, spec, mode='pct', pct_or_threshold=pct_list[i-1], kernel=kernel, n_fft=window_size, hop_size=hop_size, sr=sr, y_axis=y_axis)
            to_be_further_refined.append([spec_zoom, detect_musical_regions.musical_regions_to_ranges(indices, original_shape, x_axis, y_axis, kernel, sr=sr, hop_size=hop_size)])

        sr = 44100
        hop_size = o_hop_size
        n_fft = o_n_fft
        time_step = o_hop_size / sr
        for subregion in to_be_further_refined:
            base_spec = subregion[0]
            for ranges in subregion[1]:
                freq_range = ranges[0]
                time_range = ranges[1]
                # Compensacao da centralizacao das janelas no tempo
                time_range[0] -= time_step/2
                time_range[1] += time_step/2
                if time_range[0] < 0:
                    time_range[0] = 0

                spec_zoom, x_axis, y_axis, new_sr, window_size, hop_size = stft_zoom.stft_zoom(y, freq_range, time_range, sr=sr, original_window_size=n_fft, k=k_list[i-1])
                refined_subspec = SingleResSpectrogram(spec_zoom, x_axis, y_axis, n_fft=window_size, hop_size=hop_size, sr=new_sr)
                multires_spec.insert_zoom(base_spec, refined_subspec, freq_range, zoom_level=i, normalize=normalize)
    return multires_spec
