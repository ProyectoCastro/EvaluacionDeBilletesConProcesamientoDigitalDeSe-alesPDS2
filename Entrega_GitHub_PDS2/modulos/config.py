STANDARD_SIZE = (800, 400)

METRIC_RULES = {
    'log_Lap_Var':       {'direction': 'higher', 'weight': 2},
    'LBP_Uniformity':    {'direction': 'higher', 'weight': 2},
    'LBP_Entropy':       {'direction': 'lower',  'weight': 2},
    'GLCM_Corr':         {'direction': 'lower',  'weight': 2},
    'log_Wav_Energy':    {'direction': 'higher', 'weight': 1},
    'HF_Ratio_DWT':      {'direction': 'higher', 'weight': 1},
    'HFR':               {'direction': 'higher', 'weight': 1},
    'GLCM16_Dissim':     {'direction': 'higher', 'weight': 1},
    'GLCM_Energy':       {'direction': 'lower',  'weight': 1},
    'GLCM_Homog':        {'direction': 'lower',  'weight': 1},
    'GLCM_Contrast':     {'direction': 'lower',  'weight': 1},
    'GLCM16_Contrast':   {'direction': 'lower',  'weight': 1},
    'log_LocalVar_Mean': {'direction': 'higher', 'weight': 1},
    'log_LocalVar_Var':  {'direction': 'higher', 'weight': 2},
    'Edge_Density':      {'direction': 'higher', 'weight': 1},
    'Grad_Mean':         {'direction': 'higher', 'weight': 1},
    'Frangi':            {'direction': 'lower',  'weight': 1},
    'log_Gabor_Var':     {'direction': 'higher', 'weight': 1},
    'Lap_Kurtosis':      {'direction': 'higher', 'weight': 3},
}
