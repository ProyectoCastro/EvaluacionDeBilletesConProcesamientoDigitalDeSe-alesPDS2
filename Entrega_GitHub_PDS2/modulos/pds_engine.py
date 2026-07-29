import cv2
import numpy as np
import pywt
import json
from scipy.fftpack import fft2, fftshift
from scipy.signal import convolve2d
from scipy.ndimage import uniform_filter
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import frangi, gabor_kernel
from .config import STANDARD_SIZE

def preprocess_robusto(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: raise ValueError(f"No se pudo cargar: {image_path}")
    img_resized = cv2.resize(img_bgr, STANDARD_SIZE)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    return img_resized, gray, gray_eq

def extract_all_features(image_path):
    _, _, gray_eq = preprocess_robusto(image_path)
    
    # Frecuencia
    f_transform = fftshift(fft2(gray_eq.astype(np.float64)))
    magnitude = np.abs(f_transform)
    psd = (magnitude ** 2) / float(gray_eq.size)
    h, w = psd.shape; cy, cx = h//2, w//2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    tbin = np.bincount(r.ravel(), psd.ravel())
    nr = np.bincount(r.ravel()); nr[nr == 0] = 1
    rp = tbin / nr; cutoff = len(rp) // 4
    hfr = np.sum(rp[cutoff:]) / (np.sum(rp[:cutoff]) + 1e-10)
    
    # Gabor
    gabor_var = np.mean([convolve2d(gray_eq.astype(np.float64), np.real(gabor_kernel(0.25, theta=t, sigma_x=3.0, sigma_y=3.0)), mode='same').var() for t in [0, np.pi/4, np.pi/2, 3*np.pi/4]])
    
    # Textura
    lbp = local_binary_pattern(gray_eq, 16, 2, method='uniform')
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 19), density=True)
    hist_pos = hist_lbp[hist_lbp > 0]
    glcm8 = graycomatrix((gray_eq / 32).astype(np.uint8), distances=[1, 3], angles=[0, np.pi/4, np.pi/2], levels=8, symmetric=True, normed=True)
    glcm16 = graycomatrix((gray_eq / 16).astype(np.uint8), distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=16, symmetric=True, normed=True)
    
    # Relieve (Con Logaritmo Anti-Varianza Explosiva)
    lap = cv2.Laplacian(gray_eq, cv2.CV_64F)
    lap_flat = lap.ravel()
    lap_kurtosis = np.mean((lap_flat - lap_flat.mean())**4) / (lap_flat.var()**2 + 1e-10) - 3
    
    coeffs2 = pywt.dwt2(gray_eq.astype(np.float64), 'haar')
    LL, (LH, HL, HH) = coeffs2
    det_energy = np.sum(LH**2) + np.sum(HL**2) + np.sum(HH**2)
    
    gx, gy = cv2.Sobel(gray_eq, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(gray_eq, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    g = gray_eq.astype(np.float64)
    local_var = uniform_filter(g**2, size=5) - uniform_filter(g, size=5)**2
    
    return {
        'HFR': hfr, 'log_Gabor_Var': np.log1p(gabor_var),
        'LBP_Uniformity': np.sum(hist_lbp**2), 'LBP_Entropy': -np.sum(hist_pos * np.log2(hist_pos)),
        'GLCM_Contrast': graycoprops(glcm8, 'contrast').mean(), 'GLCM_Homog': graycoprops(glcm8, 'homogeneity').mean(),
        'GLCM_Energy': graycoprops(glcm8, 'energy').mean(), 'GLCM_Corr': graycoprops(glcm8, 'correlation').mean(),
        'GLCM16_Contrast': graycoprops(glcm16, 'contrast').mean(), 'GLCM16_Dissim': graycoprops(glcm16, 'dissimilarity').mean(),
        'log_Lap_Var': np.log1p(lap.var()), 'Lap_Kurtosis': lap_kurtosis,
        'log_Wav_Energy': np.log1p(det_energy / LH.size), 'HF_Ratio_DWT': det_energy / (np.sum(LL**2) + 1e-10),
        'Frangi': np.mean(frangi(gray_eq, sigmas=[1, 2, 3], black_ridges=True)) * 1000,
        'Edge_Density': np.sum(cv2.Canny(gray_eq, 50, 150) > 0) / gray_eq.size,
        'Grad_Mean': grad_mag.mean(), 'Grad_Std': grad_mag.std(),
        'log_LocalVar_Var': np.log1p(local_var.var()), 'log_LocalVar_Mean': np.log1p(local_var.mean())
    }

def route_face_fallback(feats, profiles_path='../perfiles_pds/'):
    """Enrutador de emergencia usando log_Lap_Var."""
    try:
        import os
        with open(os.path.join(profiles_path, 'calibracion_anti_billete20_anversonuevo.json'), 'r') as f: thresh_a = json.load(f)
        with open(os.path.join(profiles_path, 'calibracion_anti_billete20_reversonuevo.json'), 'r') as f: thresh_r = json.load(f)
    except:
        print("  [!] Error: Perfiles de Fallback no encontrados en", profiles_path)
        return None, None
    dist_a = abs(feats.get('log_Lap_Var',0) - thresh_a.get('log_Lap_Var',{}).get('mean',1))
    dist_r = abs(feats.get('log_Lap_Var',0) - thresh_r.get('log_Lap_Var',{}).get('mean',1))
    return ('billete20_anversonuevo', thresh_a) if dist_a <= dist_r else ('billete20_reversonuevo', thresh_r)
