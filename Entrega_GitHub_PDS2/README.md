# Sistema Forense Híbrido (IA + PDS) para Autenticación de Papel Moneda Peruano

## Descripción general del proyecto
Este proyecto es un pipeline computacional de nivel forense diseñado para la validación y autenticación de papel moneda peruano. Combina la capacidad de generalización semántica del Deep Learning (Inteligencia Artificial) con el rigor determinístico y la transparencia del Procesamiento Digital de Señales (PDS) en 2D. El sistema es capaz de detectar falsificaciones mediante el análisis de micro-texturas, frecuencias espaciales y relieve calcográfico (Intaglio).

## Problema que aborda
Las falsificaciones modernas de papel moneda (mediante impresión láser industrial o inyección de tinta CMYK de alta resolución) han superado las capacidades de los sistemas de escaneo óptico tradicionales basados únicamente en espectro UV/IR o reglas geométricas simples. Por otro lado, la Inteligencia Artificial pura (CNNs end-to-end) presenta un "problema de caja negra", siendo inaceptable en entornos forenses o bancarios donde se requiere explicabilidad matemática irrefutable para justificar el rechazo de un billete.

## Solución propuesta
Se propone una **Arquitectura Híbrida**:
1. **Enrutador Semántico (Deep Learning)**: Una red neuronal convolucional (EfficientNet-B0) analiza la imagen para reconocer la denominación y la cara del billete (ej. 100 Soles, Pedro Paulet, Reverso), abstrayendo rotaciones e iluminación.
2. **Motor Forense PDS (Señales en 2D)**: Una vez que la IA enruta la imagen, el sistema PDS carga el "perfil de ADN matemático" (`.json`) de ese billete específico. Posteriormente, somete la imagen a 19 algoritmos determinísticos (Filtro de Frangi, Wavelets Haar DWT, Transformada de Fourier FFT, Kurtosis Laplaciana, Patrones LBP y Matrices GLCM).
3. **Escudo Anti-CMYK y Anti-Overfitting**: Se ajustan las reglas estadísticas (Tolerancia de 3 Sigmas con compresión logarítmica) para aceptar billetes verdaderos pero muy desgastados/arrugados, mientras simultáneamente penaliza fuertemente el "ruido halftone" típico de las impresoras falsas (utilizando el alto peso de la Kurtosis Laplaciana).

## Objetivos
*   Proveer un mecanismo de autenticación transparente y auditable mediante PDS.
*   Automatizar la identificación previa de denominación y cara utilizando un enrutador IA, eliminando la necesidad de encuadres manuales perfectos.
*   Diferenciar texturas de sustrato (algodón vs bond) y relieves (Intaglio vs plano) usando exclusivamente visión por computadora.
*   Alcanzar un dictamen estadístico (Z-Score) tolerante al desgaste natural del dinero circulante.

## Limitaciones
*   **Desgaste Extremo**: Billetes supremamente viejos o que han sido lavados mecánicamente pierden por completo sus firmas de alta frecuencia (el Intaglio se aplana), lo que puede generar "Falsos Negativos".
*   **Dependencia del Set de Calibración**: La robustez del motor estadístico depende intrínsecamente del tamaño y variedad de la base de datos de calibración. El sistema necesita una base de calibración mucho más amplia y diversa (incluyendo "Clases de Desgaste") para el despliegue en producción.
*   **Ausencia de Clases Antiguas**: La IA reconoce familias de billetes antiguas, pero el motor PDS actualmente no cuenta con perfiles de calibración (`.json`) para dichas familias, lo que obliga al sistema a usar fallbacks estadísticos de emergencia.

## Detalles técnicos
*   **Sistema Operativo Compatible**: Windows / Linux / macOS
*   **Entorno de Trabajo**: Local (Jupyter Notebook / Python Virtual Environment)
*   **Lenguaje**: Python 3.10+
*   **Librerías / Dependencias Principales**:
    *   `torch` y `torchvision` (Deep Learning, Inferencia EfficientNet)
    *   `opencv-python` (cv2) (Preprocesamiento CLAHE, Laplaciano, Sobel)
    *   `scikit-image` (Extracción de GLCM, LBP, Frangi, Gabor)
    *   `scipy` (Filtros espaciales y FFT)
    *   `PyWavelets` (pywt) (Descomposición Wavelet Haar)
    *   `matplotlib` (Radiografías Visuales y Gráficos Z-Score)
    *   `numpy` (Cálculos de Tensores y Varianza Estadísitca)

## Guía de Instalación y Uso Rápido (Para Evaluadores)
Para probar el sistema sin conflictos de dependencias, recomendamos crear un entorno virtual e instalar los requerimientos exactos.

```bash
# 1. Clonar el repositorio y entrar a la carpeta
**Enlace al Dataset original**: [https://www.kaggle.com/datasets/nicolascaytuirosilva/peruvian-banknotes]
cd Entrega_GitHub_PDS2

# 2. Crear y activar un entorno virtual
python -m venv venv

# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# 3. Instalar las dependencias estrictas
pip install -r requirements.txt

# 4. Ejecutar la prueba automática visual
python test_run.py
```

> **💡 Nota sobre la Imagen de Prueba**: 
> Por defecto, el script `test_run.py` evalúa la imagen `prueba20AR.jpg`. Si deseas auditar cualquier otra imagen, simplemente abre el archivo `test_run.py` en tu editor, busca la línea 19 (`nombre_imagen = 'prueba20AR.jpg'`) y cámbiala por el nombre de cualquier otra imagen que hayas colocado dentro de la carpeta `data/test_images/`.

> **Nota**: El script `test_run.py` cargará el modelo PyTorch y evaluará una imagen de prueba mostrando en pantalla los gráficos (Radiografía PDS y Z-Scores). También puedes abrir la carpeta `notebooks/` y ejecutar `02_auditoria_hibrida_main.ipynb` celda por celda.
