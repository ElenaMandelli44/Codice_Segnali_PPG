# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:25:15 2024

@author: mandelli
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d, find_peaks
import scipy.ndimage
import scipy as xpx
import numpy as xp
import random

#%% Generation of Bids and PSF


size = 127  #127
sigma = 2



def generate_PSF(size, sigma):
    """
    Generates a Gaussian Point Spread Function (PSF) centered on one pixel.
    """
    if size % 2 == 0:
        # Dimensione pari, scegli il centro esatto
        x_center = size // 2
        y_center = size // 2
    else:
        # Dimensione dispari, scegli il centro esatto
        x_center = (size + 1) / 2
        y_center = (size + 1) / 2
    
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    
    # Calcola la distanza dall'origine (centro dell'array)
    d = np.sqrt((x - x_center)**2 + (y - y_center)**2)
    
    # Calcola la PSF gaussiana
    psf = np.exp(-d**2 / (2*sigma**2))
    
    # Normalizza la PSF in modo che la somma sia pari a 1
    psf /= np.sum(psf)
    
    return psf

# Calcola la PSF gaussiana
psf_data = generate_PSF(size, sigma)


# Visualizza le immagini generate
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 2)
plt.imshow(psf_data, cmap='hot')
plt.title('Gaussian PSF')
plt.axis('off')


plt.show()


# Generazione dell'immagine binaria
shape = (size, size)
num_dots = 100
img = np.zeros(shape, dtype=np.uint8)
coords = np.random.randint(0, min(shape), (num_dots, 2))
for coord in coords:
    img[coord[0], coord[1]] = 1

# Applica la PSF all'immagine binaria
convolved_img = convolve2d(img, psf_data, mode='same')


#%%Plotting

# Visualizza le immagini generate
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(psf_data, cmap='hot')
plt.title('Gaussian PSF')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(convolved_img, cmap='gray')
plt.title('Convolved Image with PSF')
plt.axis('off')

plt.show()

#%% Correlation

# teoricamente autocorrelazione PSF simile a Autocorrelazione Bids 

def my_correlation(function1, function2, overwrite_x=False):
    return xpx.fft.ifftshift(xpx.fft.irfftn(xp.conj(xpx.fft.rfftn(function1, overwrite_x=overwrite_x)) * xpx.fft.rfftn(function2, overwrite_x=overwrite_x), s=function1.shape, overwrite_x=overwrite_x))

Autocorrelation_PSF = my_correlation(psf_data, psf_data)
Autocorrelation_Bits = my_correlation(convolved_img, convolved_img)
Autocorrelation_image = my_correlation(img, img)
#%%

plt.figure ( figsize=(15,5))

plt.subplot(1,2,1)
plt.imshow(psf_data, cmap ="hot")
plt.title("psf")
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(Autocorrelation_PSF, cmap ="hot")
plt.title("autocorrelation psf")
plt.colorbar()

plt.show()

#%% Plotting Autocorrelation
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.imshow(Autocorrelation_PSF, cmap='gray')
plt.title('Autocorrelation PSF')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(Autocorrelation_Bits, cmap='gray')
plt.title('Autocorrelation Bits')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(Autocorrelation_image, cmap='gray')
plt.title('Autocorrelation image')
plt.colorbar()

plt.show()

#%% Extract Gaussian Regions

# Trova i picchi nell'immagine convoluta
peaks = find_peaks(convolved_img.flatten(), height=np.max(convolved_img)/3 )[0]
peak_coords = np.unravel_index(peaks, convolved_img.shape)

# Visualizza i picchi
plt.figure(figsize=(10, 10))
plt.imshow(convolved_img, cmap='gray')
plt.scatter(peak_coords[1], peak_coords[0], color='red', marker='x')
plt.title('Detected Peaks')
plt.show()

# Estrai le regioni intorno ai picchi e calcola le gaussiane
gaussian_regions = []
region_size = 10  # Dimensione della regione da estrarre intorno ai picchi

for y, x in zip(peak_coords[0], peak_coords[1]):
    if y - region_size >= 0 and y + region_size < convolved_img.shape[0] and x - region_size >= 0 and x + region_size < convolved_img.shape[1]:
        region = convolved_img[y - region_size:y + region_size + 1, x - region_size:x + region_size + 1]
        gaussian_regions.append(region)

# Visualizza le regioni gaussiane estratte
plt.figure(figsize=(15, 15))
for i, region in enumerate(gaussian_regions):
    plt.subplot(15, 15, i + 1)
    plt.imshow(region, cmap='gray')
    plt.axis('off')
plt.show()



#%% relationship between PSF and its autocorrelation 

# Calculate profiles along x and y axes
profile_x_PSF= psf_data[size // 2, :]
profile_y_PSF = psf_data[:, size // 2]

profile_x_PSF_autocorrelation= Autocorrelation_PSF[size // 2, :]
profile_y_PSF_autocorrelation = Autocorrelation_PSF[:, size // 2]

# Visualizza i profili su un grafico combinato
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(profile_x_PSF, label='PSF (x-axis)')
plt.plot(profile_x_PSF_autocorrelation, label='Autocorrelation PSF (x-axis)')
plt.title('Profiles along x-axis')
plt.xlabel('Pixel index')
plt.ylabel('Value')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(profile_y_PSF, label='PSF (y-axis)')
plt.plot(profile_y_PSF_autocorrelation, label='Autocorrelation PSF (y-axis)')
plt.title('Profiles along y-axis')
plt.xlabel('Pixel index')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()


# Calcolo di sigma e FWHM
def calculate_sigma_fwhm(profile):
    half_max = np.max(profile) / 2
    indices = np.where(profile >= half_max)[0]
    fwhm = indices[-1] - indices[0]
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return sigma, fwhm

sigma_PSF, fwhm_PSF = calculate_sigma_fwhm(profile_x_PSF)
sigma_Auto, fwhm_Auto = calculate_sigma_fwhm(profile_x_PSF_autocorrelation)

print(f'Sigma PSF: {sigma_PSF:.2f}, FWHM PSF: {fwhm_PSF:.2f}')
print(f'Sigma Autocorrelation PSF: {sigma_Auto:.2f}, FWHM Autocorrelation PSF: {fwhm_Auto:.2f}')

#%% Autocorrelation of some peaks

gaussian_regions =random.sample(gaussian_regions, 10)

# Visualizza le regioni gaussiane estratte
plt.figure(figsize=(12, 6))
for i, region in enumerate(gaussian_regions):
    plt.subplot(4, 4, i + 1)
    plt.imshow(region, cmap='gray')
    plt.axis('off')
plt.show()

# Funzione per calcolare l'autocorrelazione
def my_correlation(function1, function2, overwrite_x=False):
    return xpx.fft.ifftshift(xpx.fft.irfftn(xp.conj(xpx.fft.rfftn(function1, overwrite_x=overwrite_x)) * xpx.fft.rfftn(function2, overwrite_x=overwrite_x), s=function1.shape, overwrite_x=overwrite_x))

# Calcola l'autocorrelazione per ciascuna regione e visualizza
autocorrelations = []

plt.figure(figsize=(15, 15))
for i, region in enumerate(gaussian_regions):
    autocorr = my_correlation(region, region)
    autocorrelations.append(autocorr)
    
    plt.subplot(4, 4, i + 1)
    plt.imshow(autocorr, cmap='gray')
    plt.title(f'Autocorrelation {i+1}')
    plt.colorbar()
    plt.axis('off')

plt.show()

# Calcola la media delle autocorrelazioni
mean_autocorrelation = np.mean(autocorrelations, axis=0)


# Visualizza la media delle autocorrelazioni
plt.figure(figsize=(10, 10))
plt.imshow(mean_autocorrelation, cmap='hot')
plt.title('Mean Autocorrelation')
plt.colorbar()
plt.axis('off')
plt.show()

#%% Cropp PSF and comparison between all the profiles

# Dimensioni della regione intorno ai picchi
region_size = 10
crop_size = 2 * region_size + 1

# Cropping della PSF e dell'autocorrelazione della PSF e della media
def crop_center(img, crop_size):
    center_y, center_x = img.shape[0] // 2, img.shape[1] // 2
    start_x = center_x - (crop_size // 2)
    start_y = center_y - (crop_size // 2)
    return img[start_y:start_y + crop_size, start_x:start_x + crop_size]

cropped_psf = crop_center(psf_data, crop_size)
cropped_autocorrelation_psf = crop_center(Autocorrelation_PSF, crop_size)
cropped_mean = crop_center (mean_autocorrelation, crop_size)




#%%


# Visualizza le versioni croppate della PSF e dell'autocorrelazione della PSF
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cropped_psf, cmap='hot')
plt.title('Cropped PSF')
plt.colorbar()
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cropped_autocorrelation_psf, cmap='hot')
plt.title('Cropped Autocorrelation PSF')
plt.colorbar()
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cropped_mean, cmap='hot')
plt.title('Mean Autocorrelation')
plt.colorbar()
plt.axis('off')

plt.tight_layout()
plt.show()


#%%
# Calculate profiles along x and y axes
profile_x_PSF_C= cropped_psf[crop_size // 2, :]
profile_y_PSF_C = cropped_psf[:, crop_size// 2]

profile_x_PSF_autocorrelation_C= cropped_autocorrelation_psf[crop_size// 2, :]
profile_y_PSF_autocorrelation_C = cropped_autocorrelation_psf[:, crop_size // 2]

profile_x_PSF_mean= cropped_mean [crop_size// 2, :]
profile_y_PSF_mean =  cropped_mean[:, crop_size // 2]


# Visualizza i profili su un grafico combinato
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(profile_x_PSF_C, label='PSF (x-axis)')
plt.plot(profile_x_PSF_autocorrelation_C, label='Autocorrelation PSF (x-axis)')
plt.plot(profile_x_PSF_mean, label='mean PSF (x-axis)')
plt.title('Profiles along x-axis')
plt.xlabel('Pixel index')
plt.ylabel('Value')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(profile_y_PSF_C, label='PSF (y-axis)')
plt.plot(profile_y_PSF_autocorrelation_C, label='Autocorrelation PSF (y-axis)')
plt.plot(profile_y_PSF_mean, label='mean PSF (y-axis)')
plt.title('Profiles along y-axis')
plt.xlabel('Pixel index')
plt.ylabel('Value')

plt.legend()

plt.tight_layout()
plt.show()


# Calcolo di sigma e FWHM
def calculate_sigma_fwhm(profile):
    half_max = np.max(profile) / 2
    indices = np.where(profile >= half_max)[0]
    fwhm = indices[-1] - indices[0]
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return sigma, fwhm

sigma_PSF, fwhm_PSF = calculate_sigma_fwhm(profile_x_PSF_C)
sigma_Auto, fwhm_Auto = calculate_sigma_fwhm(profile_x_PSF_autocorrelation_C)
sigma_mean, fwhm_mean = calculate_sigma_fwhm(profile_x_PSF_mean)

print(f'Sigma PSF: {sigma_PSF:.2f}, FWHM PSF: {fwhm_PSF:.2f}')
print(f'Sigma Autocorrelation PSF: {sigma_Auto:.2f}, FWHM Autocorrelation PSF: {fwhm_Auto:.2f}')
print (f'Sigma Mean Autocorrelations PSF: {sigma_mean:.2f}, FWHM PSF {fwhm_mean:.2f}')

