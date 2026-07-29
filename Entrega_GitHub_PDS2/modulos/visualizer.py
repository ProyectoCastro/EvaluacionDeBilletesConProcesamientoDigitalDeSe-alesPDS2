import os
import cv2
import json
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
from scipy.ndimage import uniform_filter
from skimage.feature import local_binary_pattern
from skimage.filters import frangi

from .pds_engine import preprocess_robusto, extract_all_features, route_face_fallback
from .ai_router import map_pytorch_class_to_pds_folder

def visual_audit(image_path):
    """Radiografía visual 3x3."""
    img_bgr, gray, gray_eq = preprocess_robusto(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    f_transform = fftshift(fft2(gray_eq.astype(np.float64)))
    magnitude = np.log(np.abs(f_transform) + 1) 
    lbp = local_binary_pattern(gray_eq, 16, 2, method='uniform')
    lap = cv2.Laplacian(gray_eq, cv2.CV_64F)
    edges = cv2.Canny(gray_eq, 50, 150)
    coeffs2 = pywt.dwt2(gray_eq.astype(np.float64), 'haar')
    dwt_details = np.abs(coeffs2[1][0]) + np.abs(coeffs2[1][1]) + np.abs(coeffs2[1][2])
    frangi_img = frangi(gray_eq, sigmas=[1, 2, 3], black_ridges=True)
    g = gray_eq.astype(np.float64)
    local_var = uniform_filter(g**2, size=5) - uniform_filter(g, size=5)**2

    fig, axes = plt.subplots(3, 3, figsize=(22, 14))
    fig.suptitle(f"RADIOGRAFÍA PDS: {os.path.basename(image_path)}", fontsize=18, fontweight="bold")
    axes[0,0].imshow(img_rgb); axes[0,0].set_title("1. Original"); axes[0,0].axis('off')
    axes[0,1].imshow(gray_eq, cmap='gray'); axes[0,1].set_title("2. CLAHE"); axes[0,1].axis('off')
    axes[0,2].imshow(magnitude, cmap='viridis'); axes[0,2].set_title("3. Espectro FFT"); axes[0,2].axis('off')
    axes[1,0].imshow(lbp, cmap='inferno'); axes[1,0].set_title("4. LBP"); axes[1,0].axis('off')
    axes[1,1].imshow(cv2.convertScaleAbs(lap), cmap='gray'); axes[1,1].set_title("5. Laplaciano"); axes[1,1].axis('off')
    axes[1,2].imshow(edges, cmap='gray'); axes[1,2].set_title("6. Canny Edge"); axes[1,2].axis('off')
    axes[2,0].imshow(dwt_details, cmap='magma'); axes[2,0].set_title("7. DWT Wavelets"); axes[2,0].axis('off')
    axes[2,1].imshow(frangi_img, cmap='hot'); axes[2,1].set_title("8. Frangi"); axes[2,1].axis('off')
    axes[2,2].imshow(local_var, cmap='plasma'); axes[2,2].set_title("9. Varianza Local"); axes[2,2].axis('off')
    plt.tight_layout()
    plt.show()

def audit_hybrid_tolerant(image_path, router_cnn, profiles_path='../perfiles_pds/'):
    visual_audit(image_path)
    sep = '=' * 90
    print(f"\n{sep}\nAUDITORIA HÍBRIDA PDS-IA (ANTI-OVERFITTING): {os.path.basename(image_path)}\n{sep}\n")

    class_name, confidence = router_cnn.predict(image_path)
    print(f"[Fase 1: Enrutamiento Semántico PyTorch]")
    print(f">>> Clase Detectada por IA: {str(class_name).upper()} (Confianza: {confidence:.2f}%)\n")

    folder_name = map_pytorch_class_to_pds_folder(class_name) if class_name else ""
    json_file = os.path.join(profiles_path, f'calibracion_anti_{folder_name}.json')
    
    try: feats = extract_all_features(image_path)
    except Exception as e: print(f"Error: {e}"); return

    if confidence < 60.0 or not os.path.exists(json_file):
        print(f"--- ALERTA: IA con baja confianza o JSON inexistente. ACTIVANDO FALLBACK PDS ---\n")
        folder_name, thresholds = route_face_fallback(feats, profiles_path)
        if not thresholds: return
        print(f">>> Fallback matemático seleccionó perfil: {folder_name}\n")
    else:
        with open(json_file, 'r') as f: thresholds = json.load(f)

    print(f"[Fase 2: Auditoria Matemática Anti-Overfitting]")
    score, max_score, results = 0, 0, []
    for k, t in thresholds.items():
        val = feats.get(k, 0)
        weight = t['weight']
        max_score += weight
        passed = val >= t['threshold'] if t['direction'] == 'higher' else val <= t['threshold']
        z_score = abs(val - t.get('mean', 0)) / (t.get('std', 1) + 1e-10)
        if passed: 
            score += weight; icon = '[OK]'
        else: 
            icon = '[X] '
        results.append((k, val, t['threshold'], passed, icon, weight, z_score))

    print(f" {'METRICA':<22} {'EST':<6} {'VALOR':>12} {'UMBRAL EXIGIDO':>18} {'Z-SCORE':>8} {'PESO':>4}")
    print('-' * 90)
    for k, val, thr, passed, icon, weight, z in results:
        d = '>=' if thresholds[k]['direction'] == 'higher' else '<='
        print(f' {k:<22} {icon:<6} {val:>12.4f} {d:>3} {thr:<14.4f} {z:>8.2f} {weight:>4}')

    pct = (score / max_score) * 100 if max_score > 0 else 0
    print(f'\n{sep}\n PUNTAJE: {score}/{max_score} ({pct:.1f}%)\n')

    if pct >= 80: label, color = 'BILLETE AUTENTICO', '#2ecc71'
    elif pct >= 65: label, color = 'RESULTADO DUDOSO (DESGASTE)', '#f39c12'
    else: label, color = 'BILLETE FALSO', '#e74c3c'
    print(f" >>> DICTAMEN: {label}\n{sep}")

    # GRAFICO Z-SCORES
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [r[0] for r in results]
    z_scores = [r[6] for r in results]
    colors_bar = ['#2ecc71' if r[3] else '#e74c3c' for r in results]
    ax.barh(names, z_scores, color=colors_bar, edgecolor='black', linewidth=0.5)
    ax.axvline(x=3.0, color='orange', linestyle='--', linewidth=2, label='Umbral 3.0 Sigmas')
    ax.set_xlabel('Desviaciones Estándar (Z-Score) respecto al billete real')
    ax.set_title(f'Análisis de Desviación - {label}', fontsize=14, fontweight='bold', color=color)
    ax.legend(); ax.grid(axis='x', alpha=0.3); ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
