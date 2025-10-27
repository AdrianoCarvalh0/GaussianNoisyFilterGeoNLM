import numpy as np

def add_low_noise_gaussian(img):
    """
    Adds low-level Gaussian noise to a 2D grayscale image.    
    """
    m, n = img.shape                  # Image height (m) and width (n)
    sigma = 5                         # Standard deviation for low noise
    noise = np.random.normal(0, sigma, (m, n)).astype(np.float32)  # Gaussian noise ~ N(0, sigma)
    noised = np.clip(img + noise, 0, 255)                          # Clip to valid 8-bit range
    return noised                     # Returns float array (no dtype cast here)


def add_moderate_noise_gaussian(img):
    """
    Adds moderate Gaussian noise to a 2D grayscale image.    
    """
    m, n = img.shape                  # Image height (m) and width (n)
    sigma = 10                        # Standard deviation for moderate noise
    noise = np.random.normal(0, sigma, (m, n)).astype(np.float32)  # Gaussian noise ~ N(0, sigma)
    noised = np.clip(img + noise, 0, 255)                          # Clip to valid 8-bit range
    return noised                     # Returns float array (no dtype cast here)


def add_high_noise_gaussian(img):
    """
    Adds high-level Gaussian noise to a 2D grayscale image.
    """
    m, n = img.shape                  # Image height (m) and width (n)
    sigma = 15                        # Standard deviation for high noise (as given)
    noise = np.random.normal(0, sigma, (m, n)).astype(np.float32)  # Gaussian noise ~ N(0, sigma)
    noised = np.clip(img + noise, 0, 255)                          # Clip to valid 8-bit range
    return noised                     # Returns float array (no dtype cast here)