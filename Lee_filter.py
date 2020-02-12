from scipy.ndimage.filters import uniform_filter


def lee_filter(band, window, var_noise=0.25):
    # band: SAR data to be despeckled (already reshaped into image dimensions)
    # window: descpeckling filter window (tuple)
    # default noise variance = 0.25
    # assumes noise mean = 0

    mean_window = uniform_filter(band, window)
    mean_sqr_window = uniform_filter(band ** 2, window)
    var_window = mean_sqr_window - mean_window ** 2

    weights = var_window / (var_window + var_noise)
    band_filtered = mean_window + weights * (band - mean_window)
    return band_filtered