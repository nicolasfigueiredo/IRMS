import numpy as np
import scipy.stats
from librosa import amplitude_to_db
from fast_histogram import histogram1d
from irms.util import find_nearest

# This script contains functions that perform feature extraction from a spectrogram.
# The main function is calc_map_aug2(), that subdivides a spectrogram into subregions of
# dimensions given by cents x ms, and then applies a function to each subregion (mean, std deviation, entropy)

def histogram(array, bins=10):
    range = [np.min(array), np.max(array)+0.0001]
    bin_range = (range[1] - range[0]) / 10
    hist = histogram1d(array, 10, range)
    return hist / (np.sum(hist) * bin_range)

def renyi_entropy(tfp_region, alpha=3):  # without normalizaton factor
    # hist = np.histogram(tfp_region, bins=10, density=True)[0]
    hist = histogram(tfp_region, bins=10)
    return (1/(1-alpha)) * np.log2(np.sum(hist ** alpha))
    
def shannon_entropy(tfp_region):
    hist = histogram(tfp_region, bins=10)
    return scipy.stats.entropy(hist)

def find_freq_list(fft_freqs, delta_f_c):
    # Returns the frequency list that deetermines the musical interval in cents
    # Ex: between fft_freqs[idx_list[i]] and fft_freqs[idx_list[i+1]] there is an interval of delta_f_c cents
    idx_list = [0]
    if len(fft_freqs) == 1:
        return idx_list
    freq_step = fft_freqs[1] - fft_freqs[0]
    
    if fft_freqs[0] < 0.0001:   # then fft_freqs[0] == 0, estamos analisando o espectrograma inteiro
        f1 = fft_freqs[1]
    else:
        f1 = fft_freqs[0]
    
    f2 = f1 * 2 ** (delta_f_c/1200) 
    idx_f1 = 0
    idx_f2 = int(np.ceil((f2 - f1)/freq_step))
        
    while idx_f2 < len(fft_freqs):
        idx_list.append(idx_f2)
        idx_f1 = idx_f2
        f1 = fft_freqs[idx_f1]
        f2 = f1 * 2 ** (delta_f_c/1200)  
        idx_f2 = idx_f1 + int(np.ceil((f2 - f1)/freq_step))

    if idx_list[-1] != len(fft_freqs) - 1:
        idx_list.append(len(fft_freqs) - 1)
        
    return idx_list

def calc_map_aug2(spectrogram, kernel_dimensions, n_fft=2048, hop_size=512, sr=44100, type='dp', alpha=3, fft_freqs=None):
    # calc_map_aug2() divides a spectrogram into subregions of dimensions given by kernel_dimensions, where:
    #   kernel_dimensions[0] is given in ms;
    #   kernel_dimensions[1] is given in cents, and therefore is not linear in frequency.
    # After this division, a mapping is applied for each subregion according to the 'type' keyword, and the
    # resulting matrix is returned.

    if fft_freqs is None:
        idx_list = find_freq_list(fft_frequencies(sr=sr, n_fft=n_fft), kernel_dimensions[1]) # essa linha: 
    else:
        idx_list = find_freq_list(fft_freqs, kernel_dimensions[1]) # para calcular mapa de regiões refinadas, já passamos o eixo de frequências     

    delta_t_ms = kernel_dimensions[0]   # dimensão em ms

    # ms_per_frame = (n_fft+hop_size) * 1000 / (sr*2)   # discussão
    ms_per_frame = hop_size * 1000 / sr
    delta_t = int(np.round(delta_t_ms / ms_per_frame))

    mapping = np.zeros([len(idx_list)-1, spectrogram.shape[1]//delta_t])
    
    j = 0
    j_map = 0

    while j < spectrogram.shape[1] - delta_t:
        for i_map in range(len(idx_list)-1):
            subregion = spectrogram[idx_list[i_map]:idx_list[i_map+1], j:j+delta_t]
            if type=='std dev':
                mapping[i_map, j_map] = np.std(subregion)
            elif type=='shannon':
                mapping[i_map, j_map] = shannon_entropy(subregion)
            elif type=='renyi':
                mapping[i_map, j_map] = renyi_entropy(subregion, alpha=alpha)
            elif type=='var':
                mapping[i_map, j_map] = np.var(subregion)
            elif type=='dp':
                mapping[i_map, j_map] = np.sqrt(np.var(subregion))
            elif type=='avg':
                mapping[i_map, j_map] = np.mean(subregion)
            elif type=='maxmin':
                mapping[i_map, j_map] = np.max(subregion) - np.min(subregion)
        j += delta_t
        j_map += 1
    
    return mapping

def extract_features(spec_amp, kernel_dimensions, n_fft=2048, hop_size=512, sr=44100, alpha=3, fft_freqs=None):
    # A custom version of calc_map_aug2() that extracts renyi and shannon entropies at the same time

    # calc_map_aug2() divides a spectrogram into subregions of dimensions given by kernel_dimensions, where:
    #   kernel_dimensions[0] is given in ms;
    #   kernel_dimensions[1] is given in cents, and therefore is not linear in frequency.
    # After this division, a mapping is applied for each subregion according to the 'type' keyword, and the
    # resulting matrix is returned.

    if fft_freqs is None:
        idx_list = find_freq_list(fft_frequencies(sr=sr, n_fft=n_fft), kernel_dimensions[1]) # essa linha:
    else:
        idx_list = find_freq_list(fft_freqs, kernel_dimensions[1]) # para calcular mapa de regiões refinadas, já passamos o eixo de frequências

    ms_per_frame = hop_size * 1000 / sr
    delta_t = int(np.round(kernel_dimensions[0] / ms_per_frame))

    if len(idx_list) == 1:
        mapping_shannon = np.zeros([1, spec_amp.shape[1]//delta_t])
        mapping_renyi   = np.zeros([1, spec_amp.shape[1]//delta_t])
        
        j = 0
        j_map = 0

        spec_db = amplitude_to_db(spec_amp, ref=np.min)

        while j < spec_amp.shape[1] - delta_t:
            subregion_amp = spec_amp[0, j:j+delta_t]
            subregion_db  = spec_db[0, j:j+delta_t]
            mapping_shannon[0, j_map] = shannon_entropy(subregion_db)
            mapping_renyi[0, j_map] = renyi_entropy(subregion_amp, alpha=alpha)
            j += delta_t
            j_map += 1
        return mapping_shannon, mapping_renyi

    if delta_t == 0:
        mapping_shannon = np.zeros([len(idx_list)-1, 1])
        mapping_renyi   = np.zeros([len(idx_list)-1, 1])
        
        j = 0
        j_map = 0

        spec_db = amplitude_to_db(spec_amp, ref=np.min)

        for i_map in range(len(idx_list)-1):
            subregion_amp = spec_amp[idx_list[i_map]:idx_list[i_map+1], 0]
            subregion_db = spec_db[idx_list[i_map]:idx_list[i_map+1], 0]
            mapping_shannon[i_map, j_map] = shannon_entropy(subregion_db)
            mapping_renyi[i_map, j_map] = renyi_entropy(subregion_amp, alpha=alpha)
        
        return mapping_shannon, mapping_renyi

    mapping_shannon = np.zeros([len(idx_list)-1, spec_amp.shape[1]//delta_t])
    mapping_renyi   = np.zeros([len(idx_list)-1, spec_amp.shape[1]//delta_t])
    
    j = 0
    j_map = 0

    spec_db = amplitude_to_db(spec_amp, ref=np.min)

    while j < spec_amp.shape[1] - delta_t:
        for i_map in range(len(idx_list)-1):
            subregion_amp = spec_amp[idx_list[i_map]:idx_list[i_map+1], j:j+delta_t]
            subregion_db = spec_db[idx_list[i_map]:idx_list[i_map+1], j:j+delta_t]
            mapping_shannon[i_map, j_map] = shannon_entropy(subregion_db)
            mapping_renyi[i_map, j_map] = renyi_entropy(subregion_amp, alpha=alpha)
        j += delta_t
        j_map += 1
    
    return mapping_shannon, mapping_renyi

# helper function taken from librosa
def fft_frequencies(sr=44100, n_fft=2048):
    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft//2),
    endpoint=True)



## DEPRECATED
# def calc_map_aug(spectrogram, kernel_dimensions, n_fft=2048, hop_size=512, sr=44100, type='dp', alpha=3):
#     mapping = np.zeros(spectrogram.shape)
#     delta_t_ms = kernel_dimensions[0]   # dimensão em ms
#     delta_f_c = kernel_dimensions[1]   # dimensão em cents

#     ms_per_frame = (n_fft+hop_size) * 1000 / (sr*2)   # discussão
#     delta_t = int(np.ceil(delta_t_ms / ms_per_frame))
    
#     fft_freq = fft_frequencies(sr=sr, n_fft=n_fft)
#     fft_freq_step = fft_freq[1] # intervalo em Hz entre bins do spec
   
    
#     j = 0

#     while j < spectrogram.shape[1] - delta_t:
#         f1 = fft_freq[1] # começar a contar intervalo musical do 1o bin (n podemos começar de 0 hz)
#         f2 = f1 * 2 ** (delta_f_c/1200) 
#         # print(f2)
#         idx_f1 = 0
#         idx_f2 = int(np.ceil((f2)/fft_freq_step))
 
#         while idx_f2 < spectrogram.shape[0]:
#             subregion = spectrogram[idx_f1:idx_f2, j:j+delta_t]
#             if type=='dp':
#                 mapping[idx_f1:idx_f2, j:j+delta_t] = np.sqrt(np.var(subregion))
#             elif type=='avg':
#                 mapping[idx_f1:idx_f2, j:j+delta_t] = np.mean(subregion)
#             elif type=='shannon':
#                 mapping[idx_f1:idx_f2, j:j+delta_t] = shannon_entropy(subregion)
#             elif type=='renyi':
#                 mapping[idx_f1:idx_f2, j:j+delta_t] = renyi_entropy(subregion, alpha=alpha, n_fft=n_fft, sr=sr, hop_size=hop_size)

#             idx_f1 = idx_f2
#             f1 = fft_freq[idx_f1]
#             f2 = f1 * 2 ** (delta_f_c/1200)  
#             idx_f2 = idx_f1 + int(np.ceil((f2 - f1)/fft_freq_step))
#             # print(f2)
#         j += delta_t
    
#     return mapping


# def calc_var_map(spectrogram, kernel_dimensions):
#     var_map = np.zeros(spectrogram.shape)
#     delta_x = kernel_dimensions[0]
#     delta_y = kernel_dimensions[1]

#     i = 0
#     j = 0

#     while i < spectrogram.shape[0] - delta_x:
#         while j < spectrogram.shape[1] - delta_y:
#             subregion = spectrogram[i:i+delta_x, j:j+delta_y] 
#             var_map[i:i+delta_x, j:j+delta_y] = np.var(subregion) 
#             j += delta_y
#         j = 0
#         i += delta_x
    
#     return var_map

# # Calcula densidade de eventos por região retangular do mapa de eventos
# def calc_event_density(tfp_pianoroll, kernel_dimensions):
#     # tfp_pianoroll = (tfp_pianoroll > 0).astype(int)  # transforma em matriz binária (pode ser que eventos 
#                                                      # tenham sido somados ao construir o mapa de eventos (queremos isso?))
#     density = np.zeros(tfp_pianoroll.shape)

#     delta_x = kernel_dimensions[0]
#     delta_y = kernel_dimensions[1]

#     size_subreg = delta_x * delta_y

#     i = 0
#     j = 0

#     while i < tfp_pianoroll.shape[0] - delta_x:
#         while j < tfp_pianoroll.shape[1] - delta_y:
#             subregion = tfp_pianoroll[i:i+delta_x, j:j+delta_y] 
#             density[i:i+delta_x, j:j+delta_y] = np.sum(subregion.flatten())
#             j += delta_y
#         j = 0
#         i += delta_x
    
#     return density / size_subreg

# def calc_map(spectrogram, kernel_dimensions, type='dp'):
#     mapping = np.zeros(spectrogram.shape)
#     delta_x = kernel_dimensions[0]
#     delta_y = kernel_dimensions[1]

#     i = 0
#     j = 0

#     while i < spectrogram.shape[0] - delta_x:
#         while j < spectrogram.shape[1] - delta_y:
#             subregion = spectrogram[i:i+delta_x, j:j+delta_y]
#             if type=='dp':
#                 mapping[i:i+delta_x, j:j+delta_y] = np.sqrt(np.var(subregion))
#             elif type=='avg':
#                 mapping[i:i+delta_x, j:j+delta_y] = np.mean(subregion)
#             j += delta_y
#         j = 0
#         i += delta_x
    
#     return mapping
