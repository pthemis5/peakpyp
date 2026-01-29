import numpy as np
import numpy as np
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import corner
from time import time



# Define a 2D Gaussian to fit
def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, amplitude, offset):
    return offset + amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2)))



def perpare_imgfit(img):
        # perpare for fitting and make initial guesses
    x = np.arange(img.shape[1])
    y = np.arange(img.shape[0])
    x, y = np.meshgrid(x, y)
    x_flat = x.ravel()
    y_flat = y.ravel()
    img_flat = img.ravel()
    initial_guess = (np.unravel_index(np.argmax(img), img.shape)[1], np.unravel_index(np.argmax(img), img.shape)[0],  1, 1,  img.max(),  img.min())  

    return x, y, x_flat, y_flat, img_flat, initial_guess


def fit2DGaussian_scipy(img):
    x, y, x_flat, y_flat, img_flat, initial_guess = perpare_imgfit(img)

    
    params, pcov = curve_fit(lambda xy, x0, y0, sigma_x, sigma_y, amplitude, offset: gaussian_2d(xy[0], xy[1], x0, y0, sigma_x, sigma_y, amplitude, offset),
                      (x_flat, y_flat), img_flat, p0=initial_guess)

    # Extract the fitted parameters
    x0, y0, sigma_x, sigma_y, amplitude, offset = params

    # Output the center of the centroid
    centroid = (x0, y0, sigma_x, sigma_y)
    pcov_centroid = np.sqrt(np.diag(pcov)[:4])
    print("Center of the centroid:", centroid)
    print("Covariances:", pcov_centroid)
    return centroid, pcov_centroid





def fit2DGaussian_mcmc(img, centroid = None, steps = 1000, plot_mcmcstats = False, return_samples = False):


    if centroid is None:
        centroid, _ = fit2DGaussian_scipy(img)

    x, y, x_flat, y_flat, img_flat, initial_guess = perpare_imgfit(img)

    # mcmc things, also using the curve_fit result as initial guess
    def log_likelihood(params, x, y, img, noise):
        x0, y0, sigma_x, sigma_y, amplitude, offset = params
        model = gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, amplitude, offset)
        return -0.5 * np.sum(((img - model) / noise) ** 2)

    def log_prior(params):
        x0, y0, sigma_x, sigma_y, amplitude, offset = params
        if (centroid[0] - 10 < x0 < centroid[0] + 10 and
            centroid[1] - 10 < y0 < centroid[1] + 10 and
            0 < sigma_x < 2 and
            0 < sigma_y < 2 and
            0 < amplitude < 1 and
            0 < offset < 2*np.median(img)):
            return 0.0
        return -np.inf

    def log_probability(params, x, y, img, noise):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params, x, y, img, noise)
    


    noise = np.std(img)

    # Set up the MCMC sampler
    ndim = 6
    nwalkers = ndim * 5
    pos = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, img, noise))

        # Run 
    start = time()
    sampler.run_mcmc(pos, steps, progress=True)
    end = time()
    print("MCMC run time:", end - start, "seconds")
    ## print results
    samples = sampler.get_chain(discard=int(steps/2), thin=15, flat=True)

    x0_mcmc, y0_mcmc, sigma_x_mcmc, sigma_y_mcmc, amplitude_mcmc, offset_mcmc = np.median(samples, axis=0)
    pcov_mcmc = np.quantile(samples, [0.16, 0.84], axis=0)
    x0_mcmc_err, y0_mcmc_err = (pcov_mcmc[1, 0] - pcov_mcmc[0, 0])/2, (pcov_mcmc[1, 1] - pcov_mcmc[0, 1])/2

    # Output the center of the centroid
    centroid_mcmc = (x0_mcmc, y0_mcmc)
    print("Center of the centroid (MCMC):", centroid_mcmc)
    print('1 sigma CI:', x0_mcmc_err, y0_mcmc_err)


    if plot_mcmcstats:
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        labels = ["x0", "y0", "sigma_x", "sigma_y", "amplitude", "offset", 'noise']
        for i in range(ndim):
            ax = axes[i]
            ax.plot(sampler.get_chain()[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(sampler.get_chain()))
            ax.set_ylabel(labels[i])
        plt.show()

        fig_corner = corner.corner(samples, labels=labels[:-1], truths=[x0_mcmc, y0_mcmc, sigma_x_mcmc, sigma_y_mcmc, amplitude_mcmc, offset_mcmc])
        plt.show()

    if return_samples:
        raise NotImplementedError("HA HA HA if you want the samples come and get them")




    return centroid_mcmc, (x0_mcmc_err, y0_mcmc_err)


def bin_2d_array_crop(arr, bin_size=1):
    """
    Bins a 2D array into non-overlapping bins of size (bin_size x bin_size)
    by averaging the values in each bin. If the array dimensions are not
    divisible by bin_size, it crops the extra values from the bottom/right edges.

    Parameters:
    - arr: 2D numpy array of shape (H, W)
    - bin_size: size of the bin along both dimensions

    Returns:
    - binned: 2D numpy array of shape (H//bin_size, W//bin_size)
    """
    H, W = arr.shape
    H_crop = H - (H % bin_size)
    W_crop = W - (W % bin_size)
    
    cropped = arr[:H_crop, :W_crop]
    reshaped = cropped.reshape(H_crop // bin_size, bin_size, W_crop // bin_size, bin_size)
    binned = reshaped.mean(axis=(1, 3))
    return binned



def find_max_indices(arr):
    """
    Returns the indices of the maximum value in a NumPy array.

    Parameters:
    - arr: numpy array (can be 1D, 2D, etc.)

    Returns:
    - tuple of indices (e.g., (i,) for 1D, (i, j) for 2D)
    """
    flat_index = np.argmax(arr)
    return np.unravel_index(flat_index, arr.shape)


def get_centered_patch(array, center, radius):
    """
    Extract a square patch of shape (2*radius+1, 2*radius+1) centered at 'center'.
    Pads with zeros if the patch extends beyond the array borders.

    Parameters:
        array (np.ndarray): 2D array.
        center (tuple): (row, col) center of the patch.
        radius (int): Half-size of the patch (patch will be (2*radius+1)x(2*radius+1)).

    Returns:
        np.ndarray: The padded patch.
    """
    rows, cols = array.shape
    # Calculate bounds in original array
    r_start = center[0] - radius
    r_end = center[0] + radius + 1
    c_start = center[1] - radius
    c_end = center[1] + radius + 1

    # Initialize output patch with zeros

    # Calculate where to copy from the original array
    r_src_start = max(r_start, 0)
    r_src_end = min(r_end, rows)
    c_src_start = max(c_start, 0)
    c_src_end = min(c_end, cols)
    return r_src_start, c_src_start, r_src_end, c_src_end



def get_centered_patch_old(array, center, radius):
    """
    Extract a square patch of shape (2*radius+1, 2*radius+1) centered at 'center'.
    Pads with zeros if the patch extends beyond the array borders.

    Parameters:
        array (np.ndarray): 2D array.
        center (tuple): (row, col) center of the patch.
        radius (int): Half-size of the patch (patch will be (2*radius+1)x(2*radius+1)).

    Returns:
        np.ndarray: The padded patch.
    """
    rows, cols = array.shape
    patch_size = 2 * radius + 1

    # Calculate bounds in original array
    r_start = center[0] - radius
    r_end = center[0] + radius + 1
    c_start = center[1] - radius
    c_end = center[1] + radius + 1

    # Initialize output patch with zeros
    patch = np.zeros((patch_size, patch_size), dtype=array.dtype)

    # Calculate where to copy from the original array
    r_src_start = max(r_start, 0)
    r_src_end = min(r_end, rows)
    c_src_start = max(c_start, 0)
    c_src_end = min(c_end, cols)

    # Calculate where to place the source in the patch
    r_dst_start = r_src_start - r_start
    r_dst_end = r_dst_start + (r_src_end - r_src_start)
    c_dst_start = c_src_start - c_start
    c_dst_end = c_dst_start + (c_src_end - c_src_start)

    # Copy the valid region
    patch[r_dst_start:r_dst_end, c_dst_start:c_dst_end] = array[r_src_start:r_src_end, c_src_start:c_src_end]

    return patch, r_src_start, c_src_start, r_src_end, c_src_end