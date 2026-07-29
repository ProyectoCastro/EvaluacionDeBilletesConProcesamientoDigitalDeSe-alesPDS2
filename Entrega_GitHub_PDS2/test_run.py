import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ajustar ruta para encontrar la carpeta modulos
sys.path.append(os.path.join(BASE_DIR, 'notebooks'))
sys.path.append(BASE_DIR)

from modulos.ai_router import HybridRouter
from modulos.visualizer import audit_hybrid_tolerant

print("\n[1/3] Cargando Modelo de Inteligencia Artificial (EfficientNet-B0)...")
model_path = os.path.join(BASE_DIR, 'models', 'best_model_EfficientNet.pth')
router = HybridRouter(model_path=model_path)

print("\n[2/3] Iniciando Auditoría Híbrida PDS-IA...")
# Para probar cualquier otra imagen dentro de la carpeta "test_images" o la que prefiera, pero dentro de esa ruta👇
nombre_imagen = 'prueba20AR.jpg' 
imagen_sospechosa = os.path.join(BASE_DIR, 'data', 'test_images', nombre_imagen)

perfiles_path = os.path.join(BASE_DIR, 'perfiles_pds') + os.sep
audit_hybrid_tolerant(imagen_sospechosa, router, profiles_path=perfiles_path)

print("\n[3/3] ¡Auditoría completada exitosamente!")
