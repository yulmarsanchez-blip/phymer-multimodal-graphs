# -*- coding: utf-8 -*-
"""PhyMER EEG: optimización focalizada de XGBoost mediante LOSO anidado.

Conserva SIN MODIFICACIONES los métodos originales de extracción ICA, CAR,
wPLI multitaper, métricas de grafo, potencia espectral y GSP. La evaluación
usa una fila por video. El ajuste supervisado ocurre en entrenamiento interno.

NO ejecuta SVM, ExtraTrees, Elastic-Net o LDA. NO genera datos nuevos salvo
que se pida explícitamente --modo extraer o --modo todo.

El espacio ampliado incluye las ocho configuraciones XGBoost previas más
configuraciones adicionales predefinidas/aleatorizadas reproducibles.

Los resultados LOSO son exploratorios: se eligió XGBoost tras examinar
resultados externos previos sobre los mismos sujetos.
"""
from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import mne
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =====================================================================
# 1. CONFIGURACIÓN
# =====================================================================

RUTA_BASE = Path(
    r"C:\Users\Maria\Downloads\PhyMER-20250109T224509Z-001"
    r"\PhyMER\PhyMER Dataset\PhyMER Dataset\Emotiv"
)
ARCHIVO_LABELS = Path(
    r"C:\Users\Maria\Downloads\PhyMER-20250109T224509Z-001"
    r"\PhyMER\PhyMER Dataset\PhyMER Dataset\labels.csv"
)
CARPETA_SALIDA = Path(
    r"C:\Users\Maria\Downloads\Code-TG\Clasificador"
) / "comparacion_ML_GSP_5s"

SUJETOS = list(range(1, 31))
VIDEOS = list(range(1, 26))
TARGET_COLUMN = "emotion"

# Duración seleccionada preliminarmente por la evaluación metodológica.
VENTANA_SEGUNDOS = 5.0
MIN_EPOCAS_POR_REGISTRO = 5

CANALES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

BANDAS = {
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 40.0),
}
BAND_NAMES = list(BANDAS.keys())
FMIN = [BANDAS[b][0] for b in BAND_NAMES]
FMAX = [BANDAS[b][1] for b in BAND_NAMES]

# Preprocesamiento ICA (misma lógica del pipeline ya evaluado).
ICA_L_FREQ = 1.0
ICA_H_FREQ = 45.0
ICA_RANDOM_STATE = 42
ICA_EOG_THRESHOLD = 3.0

# Estimación espectral wPLI (mantener constantes en el estudio).
MT_BANDWIDTH_HZ = 2.0
MT_ADAPTIVE = False
MT_LOW_BIAS = True
N_JOBS_WPLI = 1

# Clasificación: búsqueda interna reducida y reproducible.
RANDOM_STATE = 2026
SEARCH_N_JOBS = 3
BOOTSTRAP_REPETITIONS = 2000
GRID_C = [0.1, 1.0, 10.0]
GRID_K = [60, 120, 240, "all"]

# Guardar un modelo entrenado con todos los sujetos después de evaluar LOSO.
ENTRENAR_MODELO_FINAL_TODOS = True

# Para una prueba de ejecución rápida, cambiar temporalmente SUJETOS=[1].
# Con un solo sujeto no se puede realizar la evaluación LOSO.


# =====================================================================
# 2. ARCHIVOS Y ETIQUETAS
# =====================================================================

def extraer_codigo(path: Path) -> Optional[str]:
    match = re.search(r"(SUB\d+VID\d+)", path.stem, re.IGNORECASE)
    return match.group(1).upper() if match else None


def localizar_archivo(sujeto: int, video: int) -> Optional[Path]:
    codigo = f"SUB{sujeto:02d}VID{video:02d}"
    candidates = list(RUTA_BASE.rglob(f"{codigo}*_eeg*.set"))
    if not candidates:
        candidates = [
            p for p in RUTA_BASE.rglob(f"{codigo}*.set")
            if extraer_codigo(p) == codigo
        ]
    if not candidates:
        return None
    ideal = f"{codigo}_eeg.set".lower()
    candidates = sorted(
        candidates,
        key=lambda p: (p.name.lower() != ideal, len(p.name), p.name.lower()),
    )
    if len(candidates) > 1:
        print(
            f"AVISO: {codigo}: {len(candidates)} archivos candidatos. "
            f"Se utilizará {candidates[0].name}"
        )
    return candidates[0]


def cargar_labels() -> pd.DataFrame:
    if not ARCHIVO_LABELS.exists():
        raise FileNotFoundError(f"No se encuentra labels.csv: {ARCHIVO_LABELS}")
    labels = pd.read_csv(ARCHIVO_LABELS)
    required = {"experiment_code", TARGET_COLUMN}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas {sorted(missing)} en labels.csv. "
            f"Disponibles: {list(labels.columns)}"
        )
    labels = labels.copy()
    labels["experiment_code"] = (
        labels["experiment_code"].astype(str).str.strip().str.upper()
    )
    duplicated = labels["experiment_code"].duplicated(keep=False)
    if duplicated.any():
        examples = labels.loc[duplicated, "experiment_code"].unique()[:10]
        raise ValueError(f"Códigos duplicados en labels.csv: {examples.tolist()}")
    labels[TARGET_COLUMN] = labels[TARGET_COLUMN].astype(str).str.strip()
    return labels.set_index("experiment_code", drop=False)


# =====================================================================
# 3. PREPROCESAMIENTO EEG
# =====================================================================

def preparar_eeg(ruta: Path) -> Tuple[mne.io.BaseRaw, List[int]]:
    """ICA en copia 1-45 Hz; aplicar ICA a original y CAR después."""
    raw = mne.io.read_raw_eeglab(str(ruta), preload=True, verbose=False)
    raw.pick("eeg")

    missing = [ch for ch in CANALES if ch not in raw.ch_names]
    if missing:
        raise ValueError(
            f"Faltan canales esperados en {ruta.name}: {missing}. "
            f"Canales leídos: {raw.ch_names}"
        )
    # Orden fijo: imprescindible para que cada posición de matriz tenga
    # la misma identidad de electrodo en todos los registros.
    raw.pick(CANALES)

    raw_ica = raw.copy()
    raw_ica.filter(
        l_freq=ICA_L_FREQ,
        h_freq=ICA_H_FREQ,
        fir_design="firwin",
        verbose=False,
    )

    n_components = min(len(CANALES), 14)
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=ICA_RANDOM_STATE,
        max_iter="auto",
    )
    ica.fit(raw_ica, verbose=False)

    eog_af3: List[int] = []
    eog_af4: List[int] = []
    for channel, target in (("AF3", eog_af3), ("AF4", eog_af4)):
        try:
            bads, _scores = ica.find_bads_eog(
                raw_ica,
                ch_name=channel,
                threshold=ICA_EOG_THRESHOLD,
                verbose=False,
            )
            target.extend(int(x) for x in bads)
        except Exception as exc:
            warnings.warn(
                f"No se pudieron detectar componentes EOG con {channel}: {exc}"
            )

    excluded = sorted(set(eog_af3) | set(eog_af4))
    ica.exclude = excluded

    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)
    raw_clean.set_eeg_reference(
        ref_channels="average",
        projection=False,
        verbose=False,
    )
    return raw_clean, excluded


# =====================================================================
# 4. ESTIMACIÓN wPLI
# =====================================================================

def simetrizar_wpli(matrix: np.ndarray) -> np.ndarray:
    """Devuelve una matriz simétrica, no negativa y con diagonal nula."""
    a = np.asarray(matrix, dtype=float)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(a, 0.0)
    lower = np.tril(a, k=-1)
    upper = np.triu(a, k=1)
    n_lower = np.count_nonzero(np.abs(lower) > 1e-14)
    n_upper = np.count_nonzero(np.abs(upper) > 1e-14)

    if n_lower > 0 and n_upper == 0:
        out = lower + lower.T
    elif n_upper > 0 and n_lower == 0:
        out = upper + upper.T
    elif n_lower > 1.5 * max(n_upper, 1):
        out = lower + lower.T
    elif n_upper > 1.5 * max(n_lower, 1):
        out = upper + upper.T
    else:
        out = 0.5 * (a + a.T)

    np.fill_diagonal(out, 0.0)
    out = np.clip(out, 0.0, 1.0)
    return out


def calcular_matrices_wpli(epochs: mne.Epochs) -> Dict[str, np.ndarray]:
    # Importación local: el archivo se puede inspeccionar/probar sin instalar
    # mne-connectivity; sí es obligatorio para ejecutar la extracción EEG.
    try:
        from mne_connectivity import spectral_connectivity_epochs
    except ImportError as exc:
        raise ImportError(
            "Instala mne-connectivity en el entorno de Python utilizado: "
            "pip install mne-connectivity"
        ) from exc

    con = spectral_connectivity_epochs(
        epochs,
        method="wpli",
        mode="multitaper",
        fmin=FMIN,
        fmax=FMAX,
        faverage=True,
        mt_bandwidth=MT_BANDWIDTH_HZ,
        mt_adaptive=MT_ADAPTIVE,
        mt_low_bias=MT_LOW_BIAS,
        n_jobs=N_JOBS_WPLI,
        verbose=False,
    )
    dense = con.get_data(output="dense")
    if dense.ndim != 3 or dense.shape[-1] != len(BAND_NAMES):
        raise ValueError(f"Dimensión inesperada de wPLI: {dense.shape}")
    matrices = {
        band: simetrizar_wpli(dense[:, :, idx])
        for idx, band in enumerate(BAND_NAMES)
    }
    for band, matrix in matrices.items():
        if matrix.shape != (len(CANALES), len(CANALES)):
            raise ValueError(f"Matriz {band} con dimensión inesperada: {matrix.shape}")
    return matrices


# =====================================================================
# 5. MÉTRICAS DE GRAFO: ARISTAS + RUBINOV/SPORNS + ESPECTRALES
# =====================================================================

def vector_aristas(matrix: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(matrix.shape[0], k=1)
    return np.asarray(matrix[i, j], dtype=float)


def eficiencia_global_ponderada(G: nx.Graph, nodes: Sequence[str]) -> float:
    """E_global = promedio de 1/d_ij; pares desconectados contribuyen cero.

    Para cada arista se usa length=1/weight. La longitud es matemática y
    adimensional; no representa distancia anatómica ni física.
    """
    if len(nodes) < 2:
        return 0.0
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="distance"))
    inv_distances = []
    for idx_i, u in enumerate(nodes):
        for v in nodes[idx_i + 1:]:
            d = lengths.get(u, {}).get(v, np.inf)
            inv_distances.append(1.0 / d if np.isfinite(d) and d > 0 else 0.0)
    return float(np.mean(inv_distances)) if inv_distances else 0.0


def transitividad_ponderada(A: np.ndarray) -> float:
    """Transitividad ponderada basada en intensidad geométrica de triángulos.

    Implementación de la generalización ponderada de la transitividad:
    el numerador acumula los triángulos ponderados por (w_ij*w_jk*w_ki)^(1/3)
    y el denominador corresponde a las oportunidades de triángulo binarias.
    """
    n = A.shape[0]
    binary_degree = np.count_nonzero(A > 0.0, axis=1)
    denominator = float(np.sum(binary_degree * (binary_degree - 1)))
    if denominator <= 0:
        return 0.0

    weighted_triangles_ordered = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            wij = A[i, j]
            if wij <= 0:
                continue
            for k in range(j + 1, n):
                wik = A[i, k]
                wjk = A[j, k]
                if wik > 0 and wjk > 0:
                    intensity = float(np.cbrt(wij * wik * wjk))
                    weighted_triangles_ordered += 6.0 * intensity
    return float(weighted_triangles_ordered / denominator)


def caracteristicas_grafo_banda(
    matrix: np.ndarray,
    channels: Sequence[str],
    band: str,
) -> Dict[str, float]:
    """Características de una banda. Todos los pesos wPLI se conservan.

    Medidas alineadas con Rubinov & Sporns (2010): strength, clustering
    ponderado, transitivity ponderada, global efficiency ponderada, y
    centralidades ponderadas de betweenness/closeness.

    Medidas adicionales ya consideradas en el proyecto: entropía espectral
    de laplaciana, lambda_2, Fiedler IPR y Kirchhoff.
    """
    A = np.asarray(matrix, dtype=float)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    A = np.clip(A, 0.0, 1.0)
    n = len(channels)
    if A.shape != (n, n):
        raise ValueError(f"Matriz {band} incompatible con {n} canales: {A.shape}")

    features: Dict[str, float] = {}
    upper = vector_aristas(A)
    strength = np.sum(A, axis=1)

    # Información de distribución de pesos (características ya utilizadas).
    features[f"{band}_edge_mean"] = float(np.mean(upper))
    features[f"{band}_edge_std"] = float(np.std(upper))
    features[f"{band}_edge_median"] = float(np.median(upper))
    features[f"{band}_strength_std"] = float(np.std(strength))

    # Rubinov: grado ponderado = strength. Se conserva a nivel de electrodo.
    for channel, value in zip(channels, strength):
        features[f"{band}_strength_{channel}"] = float(value)

    # Construcción del grafo ponderado no dirigido. Los pesos exactamente
    # cero se representan como ausencia de arista; no se aplica umbral.
    G = nx.Graph()
    G.add_nodes_from(channels)
    for i in range(n):
        for j in range(i + 1, n):
            weight = float(A[i, j])
            if weight > 0.0:
                G.add_edge(
                    channels[i], channels[j],
                    weight=weight,
                    distance=1.0 / max(weight, 1e-12),
                )

    # Densidad y número de componentes: controles de estructura básica.

    # Rubinov: clustering ponderado (Onnela) y estadísticos entre nodos.
    clustering = nx.clustering(G, weight="weight")
    cvals = np.asarray([clustering[ch] for ch in channels], dtype=float)
    features[f"{band}_weighted_clustering_mean"] = float(np.mean(cvals))
    features[f"{band}_weighted_clustering_std"] = float(np.std(cvals))
    features[f"{band}_weighted_clustering_max"] = float(np.max(cvals))
    # Transitivity ponderada: normalización colectiva de triángulos.
    features[f"{band}_weighted_transitivity"] = transitividad_ponderada(A)

    # Rubinov: integración mediante caminos más cortos. Se usa la longitud
    # inversa del peso; pares desconectados contribuyen cero a la eficiencia.
    features[f"{band}_global_efficiency_weighted"] = eficiencia_global_ponderada(
        G, channels
    )

    # Rubinov: centralidad de intermediación y cercanía ponderadas.
    # Se resumen entre electrodos para evitar multiplicar excesivamente la
    # dimensionalidad; la interpretación de caminos funcionales es cautelosa.
    if G.number_of_edges() > 0:
        betweenness = nx.betweenness_centrality(
            G, normalized=True, weight="distance"
        )
        closeness = nx.closeness_centrality(
            G, distance="distance", wf_improved=True
        )
        bvals = np.asarray([betweenness[ch] for ch in channels], dtype=float)
        clvals = np.asarray([closeness[ch] for ch in channels], dtype=float)
    else:
        bvals = np.zeros(n, dtype=float)
        clvals = np.zeros(n, dtype=float)

    for name, values in (("betweenness", bvals), ("closeness", clvals)):
        features[f"{band}_{name}_mean"] = float(np.mean(values))
        features[f"{band}_{name}_std"] = float(np.std(values))
        features[f"{band}_{name}_max"] = float(np.max(values))

    # Laplaciana combinatoria y sus descriptores espectrales.
    L = np.diag(strength) - A
    eigvals, eigvecs = np.linalg.eigh(L)
    eigvals = np.clip(eigvals, 0.0, None)
    tol = 1e-10 * max(float(np.max(eigvals)), 1.0)
    positive = eigvals[eigvals > tol]
    n_components_spectral = int(np.sum(eigvals <= tol))

    features[f"{band}_lambda2"] = float(eigvals[1]) if n >= 2 else 0.0

    # IPR de Fiedler: solo es identificable de forma estable si lambda_2
    # es simple; si es degenerado/desconectado, se marca como no definido.
    fiedler_defined = False
    fiedler_ipr = 0.0
    if n >= 3 and n_components_spectral == 1:
        eigengap = float(eigvals[2] - eigvals[1])
        eigengap_tol = 1e-8 * max(float(eigvals[-1]), 1.0)
        if eigengap > eigengap_tol:
            v2 = eigvecs[:, 1]
            fiedler_ipr = float(np.sum(v2 ** 4))
            fiedler_defined = True
    features[f"{band}_fiedler_ipr"] = fiedler_ipr

    # Entropía espectral con la misma normalización de los análisis
    # de fiabilidad/convergencia previos: H/log(N), no H/log(n_autovalores+).
    entropy_defined = bool(len(positive) > 0 and np.sum(positive) > tol)
    if entropy_defined:
        probabilities = positive / np.sum(positive)
        entropy_nats = float(-np.sum(probabilities * np.log(probabilities)))
        entropy_norm = entropy_nats / np.log(n)
    else:
        entropy_nats = 0.0
        entropy_norm = 0.0
    features[f"{band}_laplacian_entropy_norm"] = float(entropy_norm)

    # Índice de Kirchhoff: Kf = n * sum(1/lambda_i), válido para red conectada.
    kirchhoff_defined = bool(n_components_spectral == 1 and len(positive) == n - 1)
    kirchhoff = float(n * np.sum(1.0 / positive)) if kirchhoff_defined else 0.0
    features[f"{band}_kirchhoff"] = kirchhoff

    # Comprobación numérica de características finitas.
    for key, value in list(features.items()):
        if not np.isfinite(value):
            features[key] = 0.0
    return features


def extraer_caracteristicas_conjuntas(
    matrices: Dict[str, np.ndarray],
    channels: Sequence[str],
) -> Dict[str, float]:
    """Concatenación única de aristas + medidas de grafo para las 4 bandas."""
    features: Dict[str, float] = {}
    for band in BAND_NAMES:
        A = matrices[band]
        # 91 aristas únicas por banda: se conservan individualmente.
        i, j = np.triu_indices(len(channels), k=1)
        for ii, jj in zip(i, j):
            name = f"edge_{band}_{channels[ii]}_{channels[jj]}"
            features[name] = float(A[ii, jj])
        # Añadir métricas globales/nodales de la misma matriz.
        features.update(caracteristicas_grafo_banda(A, channels, band))
    return features



# =====================================================================
# 6. CARACTERÍSTICAS NODALES PSD + GSP
# =====================================================================

def potencia_nodal_por_banda(epochs: mne.Epochs) -> Dict[str, np.ndarray]:
    """Log-potencia integrada por banda, en dB respecto de la unidad de la
    señal que efectivamente devuelve MNE; 14 sensores x 4 bandas.
    IMPORTANTE: algunos archivos PhyMER/EPOC contienen valores ~miles aunque
    la metadata de MNE declare voltios. No llamar a estos dB físicos sin
    verificar calibración. Un factor de escala común desplaza todos los
    log-bandpowers la misma constante y desaparece al centrar nodos para GSP.

    Se estima PSD sobre las MISMAS épocas que wPLI, sin etiquetas.
    MNE devuelve PSD (época, canal, frecuencia). Primero se integra
    el espectro por banda usando trapecios; después se promedia la
    potencia lineal por época, y finalmente se transforma a dB.
    """
    spec = epochs.compute_psd(
        method="multitaper", fmin=4.0, fmax=40.0, picks=CANALES,
        bandwidth=MT_BANDWIDTH_HZ, adaptive=MT_ADAPTIVE,
        low_bias=MT_LOW_BIAS, normalization="full", remove_dc=True,
        verbose=False,
    )
    psd, freqs = spec.get_data(), np.asarray(spec.freqs)
    if psd.ndim != 3 or psd.shape[1] != len(CANALES):
        raise ValueError(f"Forma PSD inesperada: {psd.shape}")
    powers = {}
    for band, (fmin, fmax) in BANDAS.items():
        mask = (freqs >= fmin) & (freqs < fmax if band != "Gamma" else freqs <= fmax)
        if np.sum(mask) < 2:
            raise ValueError(f"Menos de 2 frecuencias para {band}")
        from scipy.integrate import trapezoid
        area = trapezoid(psd[..., mask], x=freqs[mask], axis=-1)
        power_linear = np.mean(area, axis=0)
        power_linear = np.maximum(power_linear, np.finfo(float).tiny)
        powers[band] = 10.0 * np.log10(power_linear)
    return powers


def caracteristicas_gsp_banda(A: np.ndarray, log_power_db: np.ndarray, band: str) -> Dict[str, float]:
    """GSP: señal nodal centrada x_i=10log10(PSD_banda_i)-media_nodos.

    Dirichlet: x^T L x; smoothness: Dirichlet/(x^T x).
    GFT sobre L_sym; se quita la proyección al autovector nulo
    antes de calcular distribución de energía (no es la misma
    componente constante de L cuando los strengths son distintos).
    Lowpass es suave: ganancia g(lambda)=exp(-lambda), así
    que frac_energia_filtrada=sum(exp(-2lambda)*q)/sum(q).
    Esta definición evita elegir un corte entre autovalores degenerados.
    """
    A = np.asarray(A, float)
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    x = np.asarray(log_power_db, float).copy()
    if x.shape != (n,):
        raise ValueError("Potencia nodal y grafo incompatibles")
    x -= x.mean()
    xnorm2 = float(x @ x)
    dirichlet = float(x @ L @ x)
    smoothness = dirichlet / xnorm2 if xnorm2 > 1e-18 else 0.0

    invsqrt = np.divide(1.0, np.sqrt(d), out=np.zeros_like(d), where=d > 0)
    # L_sym = D^-1/2 L D^-1/2, incluso para nodos aislados.
    Lnorm = (invsqrt[:, None] * L) * invsqrt[None, :]
    lam, U = np.linalg.eigh(Lnorm)
    lam = np.clip(lam, 0.0, 2.0)
    coeff = U.T @ x
    # Excluir el espacio de frecuencia cero, si existe.
    nonzero = lam > 1e-10
    en = coeff[nonzero] ** 2
    lam_pos = lam[nonzero]
    total = float(en.sum())
    if total > 1e-18:
        probabilities = en / total
        nonzero_p = probabilities[probabilities > 0]
        gft_entropy = float(-np.sum(nonzero_p * np.log(nonzero_p)) / np.log(max(n - 1, 2)))
        lowpass = float(np.dot(en, np.exp(-2.0 * lam_pos)) / total)
        centroid = float(np.dot(en, lam_pos) / total)
        bandwidth = float(np.sqrt(np.dot(en, (lam_pos - centroid)**2) / total))
    else:
        gft_entropy = lowpass = centroid = bandwidth = 0.0

    # Dos medidas de topología relativa separadas de GSP (Laplaciano normalizado).
    positive = lam[nonzero]
    if positive.size:
        p = positive / positive.sum()
        h_lnorm = float(-np.sum(p * np.log(p)) / np.log(n))
    else:
        h_lnorm = 0.0
    return {
        f"gsp_{band}_dirichlet": dirichlet,
        f"gsp_{band}_smoothness": smoothness,
        f"gsp_{band}_lowpass_energy_frac": lowpass,
        f"gsp_{band}_gft_entropy_norm": gft_entropy,
        f"gsp_{band}_spectral_centroid": centroid,
        f"gsp_{band}_spectral_bandwidth": bandwidth,
        f"{band}_normalized_lambda2": float(lam[1]),
        f"{band}_normalized_laplacian_entropy": h_lnorm,
    }


def extraer_caracteristicas_ampliadas(matrices, powers):
    f = extraer_caracteristicas_conjuntas(matrices, CANALES)
    for band in BAND_NAMES:
        for channel, val in zip(CANALES, powers[band]):
            f[f"bandpower_{band}_{channel}"] = float(val)
        f.update(caracteristicas_gsp_banda(matrices[band], powers[band], band))
    return f


# =====================================================================
# 7. XGBOOST ÚNICO: VALIDACIÓN ANIDADA + PERSISTENCIA REANUDABLE
# =====================================================================

import argparse
import hashlib
import itertools
import os
import platform
import shutil
import sys
from functools import partial
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.model_selection import StratifiedGroupKFold, ParameterSampler
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import recall_score


class CorrelationPruner(BaseEstimator, TransformerMixin):
    """Conserva la poda Pearson original; ajuste SOLO con X de entrenamiento."""
    def __init__(self, threshold=0.995):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X debe ser bidimensional")
        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        keep = np.ones(X.shape[1], dtype=bool)
        for i in range(X.shape[1]):
            if keep[i]:
                keep[i+1:] &= ~(np.abs(corr[i, i+1:]) > self.threshold)
        self.support_ = keep
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self):
        return self.support_.copy()


class SafeSelectKBest(BaseEstimator, TransformerMixin):
    """Mismo SelectKBest del original, ajustado por separado en cada fold."""
    def __init__(self, score_func=f_classif, k="all"):
        self.score_func = score_func
        self.k = k

    def fit(self, X, y):
        if isinstance(self.k, (int, np.integer)):
            k_real = min(int(self.k), np.asarray(X).shape[1])
        else:
            k_real = self.k
        self.selector_ = SelectKBest(score_func=self.score_func, k=k_real)
        self.selector_.fit(X, y)
        return self

    def transform(self, X):
        return self.selector_.transform(X)

    def get_support(self):
        return self.selector_.get_support()


class BalancedXGBClassifier(ClassifierMixin, BaseEstimator):
    """Estimador compatible con sklearn, con pesos calculados en .fit().

    Los parámetros nuevos reg_alpha y gamma se exponen a GridSearchCV.
    No usa validaciones externas para calcular pesos ni early stopping.
    Cada clon de XGBoost utiliza un solo hilo por defecto.
    """
    def __init__(self, max_depth=2, n_estimators=200, learning_rate=0.05,
                 min_child_weight=3.0, subsample=0.8, colsample_bytree=0.8,
                 reg_lambda=3.0, reg_alpha=0.0, gamma=0.0,
                 n_jobs=1, random_state=2026):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "Instala XGBoost en el Python de VS Code: python -m pip install xgboost"
            ) from exc
        y = np.asarray(y)
        self.classes_, local = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("XGBoost requiere al menos dos clases en el fold interno")
        freq = np.bincount(local)
        sample_weight = len(local) / (len(freq) * freq[local])
        self.model_ = XGBClassifier(
            max_depth=self.max_depth, n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            gamma=self.gamma,
            objective="multi:softprob",
            num_class=len(self.classes_),
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbosity=0,
        )
        self.model_.fit(X, local, sample_weight=sample_weight)
        return self

    def predict(self, X):
        local = np.asarray(self.model_.predict(X), dtype=int)
        return self.classes_[local]

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def feature_sets(df):
    meta = {"codigo", "sujeto", "video", "objetivo", "ventana_s", "n_epocas",
            "duracion_registro_s", "sfreq_hz", "n_componentes_ica_excluidos"}
    all_cols = sorted(c for c in df.columns
                      if c not in meta and pd.api.types.is_numeric_dtype(df[c]))
    edge = [c for c in all_cols if c.startswith("edge_")]
    bandpower = [c for c in all_cols if c.startswith("bandpower_")]
    gsp = [c for c in all_cols if c.startswith("gsp_")]
    graph = [c for c in all_cols if c not in set(edge + bandpower + gsp)]
    return {"full": all_cols, "sin_gsp": sorted(edge + graph + bandpower)}


def validar_datos(df, ventana, fs):
    required = {"codigo", "sujeto", "objetivo", "ventana_s", "n_epocas"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")
    if df.codigo.isna().any() or df.codigo.astype(str).duplicated().any():
        raise ValueError("Hay códigos de registro ausentes o duplicados")
    if (pd.to_numeric(df.ventana_s) != float(ventana)).any():
        raise ValueError("La ventana de los datos no coincide con --ventana")
    if df.objetivo.isna().any() or (df.objetivo.astype(str).str.strip() == "").any():
        raise ValueError("Faltan etiquetas emocionales")
    if df.sujeto.nunique() < 3:
        raise ValueError("LOSO requiere al menos 3 sujetos")
    subsets = feature_sets(df)
    cols = subsets[fs]
    if not cols or not any(c.startswith("edge_") for c in cols):
        raise ValueError("Este CSV no parece ser la tabla ampliada wPLI + GSP")
    if not {f"gsp_{b}_dirichlet" for b in BAND_NAMES}.issubset(df.columns):
        raise ValueError("Faltan características GSP. Usa el CSV de la extracción de 5 modelos")
    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isinf(X).any():
        raise ValueError("Se encontraron valores infinitos en las características")
    if np.isnan(X).all(axis=0).any():
        raise ValueError("Al menos una característica es completamente NaN")
    print(f"Datos: {len(df)} videos, {df.sujeto.nunique()} sujetos, "
          f"{df.objetivo.nunique()} emociones, {len(cols)} características ({fs}).")
    return X, cols


def nombres_seleccionados(pipe, names):
    names = np.asarray(names)
    names = names[pipe.named_steps["variance"].get_support()]
    names = names[pipe.named_steps["pruner"].get_support()]
    return names[pipe.named_steps["select"].get_support()].tolist()


def pipeline_xgb(selector, scaler, hilos_xgb):
    if selector == "anova":
        func = f_classif
    elif selector == "mi":
        func = partial(mutual_info_classif, random_state=RANDOM_STATE)
    else:
        raise ValueError("selector desconocido")
    scaling = StandardScaler() if scaler == "standard" else RobustScaler()
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold(threshold=1e-12)),
        ("pruner", CorrelationPruner(threshold=0.995)),
        ("scaler", scaling),
        ("select", SafeSelectKBest(score_func=func, k="all")),
        ("clf", BalancedXGBClassifier(n_jobs=hilos_xgb, random_state=RANDOM_STATE)),
    ])


def base_params(k, depth, n_trees):
    """Candidato ORIGINAL: parámetros fijos idénticos a los 5 modelos."""
    return {
        "select__k": k,
        "clf__max_depth": depth,
        "clf__n_estimators": n_trees,
        "clf__learning_rate": 0.05,
        "clf__min_child_weight": 3.0,
        "clf__subsample": 0.8,
        "clf__colsample_bytree": 0.8,
        "clf__reg_lambda": 3.0,
        "clf__reg_alpha": 0.0,
        "clf__gamma": 0.0,
    }


def candidate_key(candidate):
    return json.dumps(candidate, sort_keys=True, ensure_ascii=False)


def construir_candidatos(n_total=20, seed=RANDOM_STATE):
    """8 originales + nuevas combinaciones reproducibles. Sin duplicados.

    Basales: [120,all] × profundidad [2,3] × árboles [150,300].
    Exploración ampliada: tasa, hojas, submuestreo, columnas, L1, L2 y gamma.
    Mismos candidatos en LOS 30 folds externos: comparación transparente.
    """
    if n_total < 8:
        raise ValueError("--candidatos debe ser >=8 para conservar todas las bases originales")
    old = [base_params(k, depth, trees)
           for k, depth, trees in itertools.product([120, "all"], [2, 3], [150, 300])]
    if n_total == 8:
        return old
    # Casos deliberados para explorar expresamente el aprendizaje y la complejidad;
    # no se derivan de puntuaciones LOSO de esta ejecución.
    anchor = [
        dict(base_params("all", 1, 150)),
        dict(base_params("all", 2, 240), **{}),
        dict(base_params(120, 2, 150)),
        dict(base_params("all", 2, 150)),
    ]
    anchor[1]["clf__learning_rate"] = 0.025
    anchor[2].update({"clf__min_child_weight": 6.0, "clf__subsample": 0.7,
                      "clf__colsample_bytree": 0.6, "clf__reg_lambda": 10.0,
                      "clf__reg_alpha": 0.1, "clf__gamma": 0.5})
    anchor[3].update({"clf__learning_rate": 0.08, "clf__min_child_weight": 1.0,
                      "clf__subsample": 1.0, "clf__colsample_bytree": 1.0,
                      "clf__reg_lambda": 1.0, "clf__reg_alpha": 1.0,
                      "clf__gamma": 1.0})
    candidates = list(old)
    seen = {candidate_key(c) for c in candidates}
    for c in anchor:
        if len(candidates) >= n_total:
            break
        if candidate_key(c) not in seen:
            candidates.append(c)
            seen.add(candidate_key(c))
    if len(candidates) >= n_total:
        return candidates
    distributions = {
        "select__k": [120, "all"],
        "clf__max_depth": [1, 2, 3],
        "clf__n_estimators": [80, 150, 240, 300],
        "clf__learning_rate": [0.025, 0.05, 0.08],
        "clf__min_child_weight": [1.0, 3.0, 6.0],
        "clf__subsample": [0.7, 0.9, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__reg_lambda": [1.0, 3.0, 10.0],
        "clf__reg_alpha": [0.0, 0.1, 1.0],
        "clf__gamma": [0.0, 0.5, 1.0],
    }
    # ParameterSampler muestrea sin reemplazo cuando todos los valores son listas.
    sampled = ParameterSampler(distributions, n_iter=n_total + 32, random_state=seed)
    for sampled_candidate in sampled:
        c = dict(sampled_candidate)
        key = candidate_key(c)
        if key not in seen:
            candidates.append(c)
            seen.add(key)
        if len(candidates) >= n_total:
            break
    if len(candidates) != n_total:
        raise RuntimeError("No fue posible crear el número solicitado de candidatos")
    return candidates


def splits_internos(y, groups, inner_folds):
    if inner_folds == 0:
        splitter = LeaveOneGroupOut()
    else:
        splitter = StratifiedGroupKFold(
            n_splits=min(inner_folds, len(np.unique(groups))),
            shuffle=True, random_state=RANDOM_STATE,
        )
    xx = np.zeros((len(y), 1))
    splits = list(splitter.split(xx, y, groups))
    if any(len(np.unique(y[tr])) < 2 for tr, _ in splits):
        raise ValueError("Hay un fold interno con menos de dos clases de entrenamiento")
    return splits


def puntaje_f1_fijo(estimador, X, y, labels):
    """Denominador de clases fijo y serializable en procesos de Windows."""
    return f1_score(y, estimador.predict(X), labels=labels,
                    average="macro", zero_division=0)


def metricas(y, pred, labels):
    return {
        "macro_f1_7clases": float(f1_score(y, pred, labels=labels,
                                         average="macro", zero_division=0)),
        "balanced_accuracy_7clases": float(recall_score(y, pred, labels=labels,
                                             average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
    }


def sha256_archivo(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_configuracion(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def guardar_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                     default=str), encoding="utf-8")


def generar_manifest(csv, df, fs, args, candidates):
    import sklearn
    import xgboost
    return {
        "version_script": "2.0.0-xgb-only",
        "sha256_codigo": sha256_archivo(Path(__file__)),
        "sha256_csv": sha256_archivo(csv),
        "n_registros": len(df),
        "n_sujetos": df.sujeto.nunique(),
        "clases": sorted(df.objetivo.astype(str).unique().tolist()),
        "ventana_segundos": args.ventana,
        "feature_set": fs,
        "selector": args.selector,
        "escalador": args.scaler,
        "inner_folds": args.inner_folds,
        "semilla": RANDOM_STATE,
        "candidatos": candidates,
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
        "python": sys.version.split()[0],
        "n_jobs_busqueda": args.n_jobs,
        "hilos_por_xgboost": args.hilos_xgb,
        "protocolo": "LOSO externo + StratifiedGroupKFold interno; "
                     "f1_macro_7clases; pesos balanceados SOLO en fit",
    }


def validar_reanudacion(carpeta, firma, sujeto, n_esperado):
    info_path = carpeta / "metricas.json"
    pred_path = carpeta / "predicciones.csv"
    search_path = carpeta / "busqueda.csv"
    selected_path = carpeta / "seleccion.csv"
    if not all(p.exists() for p in [info_path, pred_path, search_path, selected_path]):
        return False
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("firma") != firma:
        raise RuntimeError(
            f"Checkpoint del sujeto {sujeto} incompatible con el CSV/código/espacio "
            "de búsqueda. Usa otra carpeta --salida o elimina sus checkpoints."
        )
    preds = pd.read_csv(pred_path)
    if len(preds) != n_esperado or preds.codigo.duplicated().any():
        raise RuntimeError(f"Checkpoint del sujeto {sujeto} incompleto o inconsistente")
    return True


def guardar_fold_atomicamente(destino, pred_df, info, search_df, selected_df):
    temporal = destino.parent / ("_incompleto_" + destino.name)
    if temporal.exists():
        shutil.rmtree(temporal)
    temporal.mkdir(parents=True)
    pred_df.to_csv(temporal / "predicciones.csv", index=False)
    search_df.to_csv(temporal / "busqueda.csv", index=False)
    selected_df.to_csv(temporal / "seleccion.csv", index=False)
    guardar_json(temporal / "metricas.json", info)
    # Directorio final solamente aparece después de escribir los cuatro archivos.
    os.replace(temporal, destino)


def juntar_checkpoints(checkpoint_dir, sujetos_esperados):
    pred, metrics_rows, search_rows, selected_rows = [], [], [], []
    for sujeto in sujetos_esperados:
        folder = checkpoint_dir / f"sujeto_{int(sujeto):02d}"
        if not folder.is_dir():
            continue
        pred.append(pd.read_csv(folder / "predicciones.csv"))
        info = json.loads((folder / "metricas.json").read_text(encoding="utf-8"))
        metrics_rows.append({k: v for k, v in info.items() if k not in ("firma",)})
        search_rows.append(pd.read_csv(folder / "busqueda.csv"))
        selected_rows.append(pd.read_csv(folder / "seleccion.csv"))
    result = (
        pd.concat(pred, ignore_index=True) if pred else pd.DataFrame(),
        pd.DataFrame(metrics_rows),
        pd.concat(search_rows, ignore_index=True) if search_rows else pd.DataFrame(),
        pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame(),
    )
    return result


def bootstrap_sujetos(p, n_boot=1000, p_ref=None):
    subjects = np.unique(p.sujeto.to_numpy())
    labels = np.arange(N_CLASES_GLOBAL)
    grouped = {s: p[p.sujeto == s] for s in subjects}
    ref_grouped = None
    if p_ref is not None:
        ref_grouped = {s: p_ref[p_ref.sujeto == s] for s in subjects}
    rg = np.random.default_rng(RANDOM_STATE)
    scores, deltas = [], []
    for _ in range(n_boot):
        chosen = rg.choice(subjects, len(subjects), replace=True)
        sample = pd.concat([grouped[s] for s in chosen], ignore_index=True)
        tr = sample.real_code.to_numpy()
        yp = sample.pred_code.to_numpy()
        scores.append(f1_score(tr, yp, labels=labels, average="macro", zero_division=0))
        if ref_grouped is not None:
            sample_ref = pd.concat([ref_grouped[s] for s in chosen], ignore_index=True)
            deltas.append(scores[-1] - f1_score(sample_ref.real_code,
                        sample_ref.pred_code, labels=labels,
                        average="macro", zero_division=0))
    row = {
        "bootstrap_n": n_boot,
        "f1_ci95_low": float(np.quantile(scores, .025)),
        "f1_ci95_high": float(np.quantile(scores, .975)),
    }
    if deltas:
        row.update({
            "delta_f1_ci95_low": float(np.quantile(deltas, .025)),
            "delta_f1_ci95_high": float(np.quantile(deltas, .975)),
        })
    return row


def cargar_referencia(path, df, fs, label_encoder):
    """Usar OOF previas XGBoost del programa original, NUNCA ajustarlas aquí."""
    reference = pd.read_csv(path)
    necessary = {"codigo", "sujeto", "objetivo_real", "objetivo_predicho"}
    if not necessary.issubset(reference.columns):
        raise ValueError(f"CSV referencia sin columnas {sorted(necessary-set(reference.columns))}")
    if "modelo" in reference:
        reference = reference[reference.modelo.astype(str).str.lower() == "xgboost"]
    if "feature_set" in reference:
        reference = reference[reference.feature_set.astype(str) == fs]
    if reference.empty:
        raise ValueError(f"Referencia sin predicciones XGBoost para feature_set={fs}")
    if reference.codigo.duplicated().any():
        raise ValueError("Referencia duplicada: requiere una fila por codigo")
    reference = reference.copy()
    truth = df.set_index("codigo")
    if set(reference.codigo.astype(str)) != set(truth.index.astype(str)):
        raise ValueError("Referencia y tabla actual contienen códigos diferentes")
    reference = reference.set_index("codigo").loc[truth.index].reset_index()
    if not np.array_equal(reference.objetivo_real.astype(str).to_numpy(),
                          truth.objetivo.astype(str).to_numpy()):
        raise ValueError("Las etiquetas de la referencia no coinciden con el dataset")
    if not np.array_equal(reference.sujeto.astype(int).to_numpy(),
                          truth.sujeto.astype(int).to_numpy()):
        raise ValueError("La distribución de sujetos de la referencia no coincide")
    reference["real_code"] = label_encoder.transform(reference.objetivo_real)
    reference["pred_code"] = label_encoder.transform(reference.objetivo_predicho)
    return reference.reset_index(drop=True)


def reportar_resultados(output, check_dir, todos, feature_set, args, df, enc):
    p, f, history, sel = juntar_checkpoints(check_dir, todos)
    if p.empty:
        return None
    p.to_csv(output / "predicciones_xgboost_LOSO.csv", index=False)
    f.to_csv(output / "metricas_xgboost_por_sujeto.csv", index=False)
    history.to_csv(output / "ajuste_interno_xgboost.csv", index=False)
    sel.to_csv(output / "caracteristicas_seleccionadas_xgboost.csv", index=False)
    completed = p.sujeto.nunique()
    print(f"Checkpoints consolidados: {completed}/{len(todos)} sujetos.")
    if completed != len(todos):
        print("ADVERTENCIA: evaluación PARCIAL. No se calcularán IC ni rendimiento LOSO final.")
        return None
    if len(p) != len(df) or set(p.codigo.astype(str)) != set(df.codigo.astype(str)):
        raise RuntimeError("Las predicciones no cubren exactamente todos los registros")
    p = p.sort_values(["sujeto", "codigo"]).reset_index(drop=True)
    p.to_csv(output / "predicciones_xgboost_LOSO.csv", index=False)
    label_codes = np.arange(len(enc.classes_))
    m = metricas(p.real_code, p.pred_code, label_codes)
    res = {
        "modelo": "xgboost_optimizado", "feature_set": feature_set,
        "n_sujetos": completed, "n_registros": len(p),
        "fold_macro_f1_mean": float(f.macro_f1_7clases.mean()),
        "fold_macro_f1_sd": float(f.macro_f1_7clases.std(ddof=1)),
        "fold_balanced_accuracy_mean": float(f.balanced_accuracy_7clases.mean()),
        **m,
    }
    reference = None
    if args.referencia:
        reference = cargar_referencia(args.referencia, df, feature_set, enc)
        reference = reference.set_index("codigo").loc[p.codigo].reset_index()
        ref_f1 = metricas(reference.real_code, reference.pred_code, label_codes)["macro_f1_7clases"]
        res["macro_f1_xgboost_anterior"] = ref_f1
        res["delta_f1_vs_anterior"] = m["macro_f1_7clases"] - ref_f1
        comparison = p[["codigo","sujeto","objetivo_real","objetivo_predicho"]].copy()
        comparison = comparison.rename(columns={"objetivo_predicho":"pred_nuevo"})
        comparison["pred_anterior"] = reference.objetivo_predicho.to_numpy()
        comparison["acierto_nuevo"] = (comparison.objetivo_real == comparison.pred_nuevo)
        comparison["acierto_anterior"] = (comparison.objetivo_real == comparison.pred_anterior)
        comparison.to_csv(output / "predicciones_pareadas_vs_anterior.csv", index=False)
    res.update(bootstrap_sujetos(p, n_boot=args.bootstrap, p_ref=reference))
    pd.DataFrame([res]).to_csv(output / "resumen_xgboost_optimizado.csv",index=False)
    cm = confusion_matrix(p.real_code,p.pred_code,labels=label_codes)
    pd.DataFrame(cm, index=["real_"+v for v in enc.classes_],
                 columns=["pred_"+v for v in enc.classes_]).to_csv(
                     output / "matriz_confusion_xgboost.csv")
    report = classification_report(p.real_code, p.pred_code,
        labels=label_codes, target_names=enc.classes_, output_dict=True,
        zero_division=0)
    guardar_json(output / "reporte_por_emocion.json", report)
    print("\n========== RESULTADO LOSO FINAL ==========")
    print(pd.DataFrame([res]).to_string(index=False))
    print("AVISO: la elección de XGBoost proviene de explorar el LOSO anterior; "
          "esta comparación es exploratoria, no una confirmación independiente.")
    return res


def evaluar_xgboost(df, csv, output, args):
    global N_CLASES_GLOBAL
    X, cols = validar_datos(df, args.ventana, args.feature_set)
    groups = df.sujeto.astype(int).to_numpy()
    enc = LabelEncoder()
    y = enc.fit_transform(df.objetivo.astype(str))
    N_CLASES_GLOBAL = len(enc.classes_)
    if N_CLASES_GLOBAL != 7:
        warnings.warn(f"Se detectaron {N_CLASES_GLOBAL} clases; se esperaba 7")
    candidates = construir_candidatos(args.candidatos)
    manifest = generar_manifest(csv, df, args.feature_set, args, candidates)
    # Hilos o workers son detalles de rendimiento: su cambio no invalida checkpoints.
    reproducibility = {k:v for k,v in manifest.items()
        if k not in {"n_jobs_busqueda", "hilos_por_xgboost"}}
    signature = hash_configuracion(reproducibility)
    experiment = output / f"experimento_{args.feature_set}"
    experiment.mkdir(parents=True, exist_ok=True)
    manifest_path = experiment / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("firma") != signature:
            raise RuntimeError("La carpeta --salida ya contiene un experimento de otra "
                               "configuración. Elige otra --salida para no mezclar folds.")
    manifest["firma"] = signature
    guardar_json(manifest_path, manifest)
    guardar_json(experiment / "candidatos.json", candidates)
    chk = experiment / "checkpoints_por_sujeto"
    chk.mkdir(exist_ok=True)
    outer = list(LeaveOneGroupOut().split(X, y, groups))
    all_subjects = [int(groups[te][0]) for _,te in outer]
    if args.max_sujetos:
        outer_todo = outer[:args.max_sujetos]
    else:
        outer_todo = outer
    grid = [{name: [val] for name,val in candidate.items()}
            for candidate in candidates]
    print("Búsqueda XGBoost exclusiva:", len(candidates), "configuraciones ×",
          (args.inner_folds if args.inner_folds else "LOSO"), "folds internos ×",
          len(outer), "sujetos externos")
    print("Paralelismo: GridSearchCV",args.n_jobs,"procesos; XGBoost",
          args.hilos_xgb,"hilo(s) por ajuste")
    print("Salida:",experiment)
    t0 = time.perf_counter()
    for fold_num, (train_idx, test_idx) in enumerate(outer_todo, start=1):
        subject = int(groups[test_idx][0])
        checkpoint = chk / f"sujeto_{subject:02d}"
        if checkpoint.exists():
            if validar_reanudacion(checkpoint, signature, subject, len(test_idx)):
                print(f"  [{fold_num:02d}/{len(outer)}] sujeto {subject:02d} "
                      "COMPLETADO previamente; se omite.",flush=True)
                continue
            raise RuntimeError(f"Checkpoint incompleto: {checkpoint}. "
                               "Retíralo manualmente si quieres recalcularlo.")
        start = time.perf_counter()
        inner = splits_internos(y[train_idx],groups[train_idx],args.inner_folds)
        pipe = pipeline_xgb(args.selector,args.scaler,args.hilos_xgb)
        search = GridSearchCV(pipe,grid,scoring=partial(puntaje_f1_fijo, labels=tuple(range(N_CLASES_GLOBAL))),cv=inner,
                              n_jobs=args.n_jobs,pre_dispatch=args.n_jobs,
                              refit=True,error_score="raise",
                              return_train_score=False,verbose=0)
        search.fit(X[train_idx],y[train_idx])
        prediction = search.best_estimator_.predict(X[test_idx])
        selected = nombres_seleccionados(search.best_estimator_,cols)
        score_row = metricas(y[test_idx],prediction,np.arange(N_CLASES_GLOBAL))
        record = {
            "firma": signature, "modelo": "xgboost", "feature_set": args.feature_set,
            "fold": int(fold_num), "sujeto": subject, "n_registros":len(test_idx),
            "n_train":len(train_idx), "n_feats_input":len(cols),
            "n_feats_selected":len(selected), "cv_macro_f1_interno":float(search.best_score_),
            "best_params":json.dumps(search.best_params_,sort_keys=True,ensure_ascii=False),
            "tiempo_segundos":float(time.perf_counter()-start),
            **score_row,
        }
        preds = pd.DataFrame({
            "modelo":"xgboost", "feature_set":args.feature_set,
            "sujeto":groups[test_idx].astype(int),
            "codigo":df.iloc[test_idx].codigo.astype(str).to_numpy(),
            "objetivo_real":enc.classes_[y[test_idx]],
            "objetivo_predicho":enc.classes_[prediction],
            "real_code":y[test_idx],"pred_code":prediction,
        })
        innerlog = pd.DataFrame([{
            "sujeto": subject, "fold":fold_num,"candidato":j+1,
            "params":json.dumps(search.cv_results_["params"][j],sort_keys=True,
                                 ensure_ascii=False),
            "mean_inner_f1":float(search.cv_results_["mean_test_score"][j]),
            "std_inner_f1":float(search.cv_results_["std_test_score"][j]),
            "mean_fit_time_s":float(search.cv_results_["mean_fit_time"][j]),
            "rank":int(search.cv_results_["rank_test_score"][j]),
        } for j in range(len(search.cv_results_["params"]))])
        sel_df = pd.DataFrame({"sujeto":subject,"fold":fold_num,
                               "caracteristica":selected})
        guardar_fold_atomicamente(checkpoint, preds, record, innerlog, sel_df)
        print(f"  [{fold_num:02d}/{len(outer)}] sujeto {subject:02d} | "
              f"F1 externo={record['macro_f1_7clases']:.4f} | "
              f"F1 interno={search.best_score_:.4f} | "
              f"k={search.best_params_['select__k']} | "
              f"{record['tiempo_segundos']:.1f}s",flush=True)
    total = time.perf_counter()-t0
    print(f"Tiempo total de la sesión: {total/60:.1f} min")
    result = reportar_resultados(experiment, chk, all_subjects, args.feature_set, args, df, enc)
    if result is not None and args.ajustar_final:
        ajustar_final_xgboost(X, y, groups, cols, enc, candidates, experiment, args)
    elif args.ajustar_final:
        print("--ajustar-final omitido: LOSO externo incompleto.")
    return result


def ajustar_final_xgboost(X, y, groups, cols, enc, candidates, output, args):
    """Modelo opcional en todos los sujetos. NO es otra evaluación independiente."""
    print(f"Ajustando modelo de despliegue sobre {len(np.unique(groups))} sujetos, sin prueba externa...",flush=True)
    inner = splits_internos(y, groups, args.inner_folds)
    pipe = pipeline_xgb(args.selector,args.scaler,args.hilos_xgb)
    grid = [{key: [value] for key,value in candidate.items()} for candidate in candidates]
    search = GridSearchCV(
        pipe,grid,scoring=partial(puntaje_f1_fijo,labels=tuple(range(len(enc.classes_)))),
        cv=inner,n_jobs=args.n_jobs,pre_dispatch=args.n_jobs,
        refit=True,error_score="raise",return_train_score=False,
    )
    search.fit(X,y)
    # Cloudpickle permite cargar el modelo aunque esta clase se haya definido
    # en un programa ejecutado como __main__ (caso habitual en Windows).
    from joblib.externals import cloudpickle
    path = output / "modelo_final_xgboost_ajustado_todos_sujetos.pkl"
    payload = {
        "model":search.best_estimator_,
        "features":cols,
        "classes":enc.classes_.tolist(),
        "best_params":search.best_params_,
        "inner_macro_f1":float(search.best_score_),
        "preprocessing":"ICA 1-45Hz copy; aplicar a raw, CAR; 5s; wPLI multitaper",
        "window_seconds":args.ventana,
        "not_evaluation":"Modelo entrenado con todos los sujetos; sin muestra externa independiente",
    }
    with path.open("wb") as fh:
        cloudpickle.dump(payload, fh)
    print("Modelo de despliegue guardado:",path)


def extraer_dataset_como_antes():
    """Se conserva el mismo método de extracción del comparador original."""
    if not RUTA_BASE.exists():
        raise FileNotFoundError(f"No existe la carpeta Emotiv: {RUTA_BASE}")
    labels = cargar_labels()
    rows, states = [], []
    for pos, (subject, video) in enumerate(itertools.product(SUJETOS, VIDEOS), start=1):
        code = f"SUB{subject:02d}VID{video:02d}"
        ruta = localizar_archivo(subject, video)
        if ruta is None:
            states.append({"codigo":code,"estado":"SIN_ARCHIVO"})
            continue
        if code not in labels.index:
            states.append({"codigo":code,"estado":"SIN_ETIQUETA"})
            continue
        target = labels.loc[code,TARGET_COLUMN]
        if pd.isna(target) or str(target).strip()=="":
            states.append({"codigo":code,"estado":"ETIQUETA_VACIA"})
            continue
        print(f"EXTRACCIÓN [{pos}/{len(SUJETOS)*len(VIDEOS)}] {code}",flush=True)
        try:
            raw, excluded = preparar_eeg(ruta)
            epochs = mne.make_fixed_length_epochs(
                raw, duration=VENTANA_SEGUNDOS, overlap=0.,preload=True,
                reject_by_annotation=True,verbose=False,
            )
            if len(epochs) < MIN_EPOCAS_POR_REGISTRO:
                states.append({"codigo":code,"estado":"EPOCAS_INSUFICIENTES",
                               "n_epocas":len(epochs)})
                continue
            features = extraer_caracteristicas_ampliadas(
                calcular_matrices_wpli(epochs), potencia_nodal_por_banda(epochs))
            rows.append(dict(codigo=code,sujeto=subject,video=video,
                objetivo=str(target),ventana_s=VENTANA_SEGUNDOS,
                n_epocas=len(epochs),duracion_registro_s=float(raw.times[-1]),
                sfreq_hz=float(raw.info["sfreq"]),
                n_componentes_ica_excluidos=len(excluded),**features))
            states.append({"codigo":code,"estado":"OK","n_epocas":len(epochs),
                           "componentes_ica_excluidos":str(excluded)})
        except Exception as exc:
            states.append({"codigo":code,
                           "estado":f"ERROR {type(exc).__name__}: {exc}"})
            print("ERROR",code,type(exc).__name__,exc,flush=True)
    if not rows:
        raise RuntimeError("No se obtuvieron registros. Comprueba estados_extraccion.csv")
    return pd.DataFrame(rows), pd.DataFrame(states)


def main():
    global RUTA_BASE, ARCHIVO_LABELS, CARPETA_SALIDA
    global VENTANA_SEGUNDOS, MIN_EPOCAS_POR_REGISTRO
    previous_output = CARPETA_SALIDA
    parser = argparse.ArgumentParser(
        description="Solo XGBoost: optimización ampliada, LOSO anidado y checkpoint por sujeto")
    parser.add_argument("--modo", choices=["evaluar","extraer","todo"],
                        default="evaluar")
    parser.add_argument("--ruta",type=Path,default=RUTA_BASE)
    parser.add_argument("--labels",type=Path,default=ARCHIVO_LABELS)
    parser.add_argument("--csv",type=Path,default=None,
        help="CSV ya extraído por el comparador original; evita reprocesar EEG")
    parser.add_argument("--salida",type=Path,default=previous_output / "XGBoost_optimizado")
    parser.add_argument("--ventana",type=float,choices=[3.0,5.0],default=5.0)
    parser.add_argument("--min-epocas",type=int,default=5)
    parser.add_argument("--feature-set",choices=["full","sin_gsp"],default="full")
    parser.add_argument("--selector",choices=["anova","mi"],default="anova")
    parser.add_argument("--scaler",choices=["standard","robust"],default="standard")
    parser.add_argument("--candidatos",type=int,default=20,
        help="Total de configuraciones: 8 originales + exploración ampliada (default 20)")
    parser.add_argument("--inner-folds",type=int,default=5,
        help="5 folds estratificados por grupo; 0=LOSO interno exhaustivo")
    parser.add_argument("--n-jobs",type=int,default=3,
        help="Procesos simultáneos de GridSearchCV (default 3)")
    parser.add_argument("--hilos-xgb",type=int,default=1,
        help="Hilos por modelo XGBoost; no multiplicar innecesariamente")
    parser.add_argument("--bootstrap",type=int,default=1000)
    parser.add_argument("--max-sujetos",type=int,default=0,
        help="Solo para comprobar ejecución; 0=los 30 sujetos")
    parser.add_argument("--referencia",type=Path,default=None,
        help="predicciones_modelos_LOSO.csv anterior; comparación pareada exploratoria")
    parser.add_argument("--ajustar-final",action="store_true",
        help="Tras completar LOSO, reajustar XGBoost con todos los sujetos y guardar .pkl")
    args = parser.parse_args()
    if args.n_jobs < 1 or args.hilos_xgb < 1:
        parser.error("--n-jobs y --hilos-xgb deben ser >= 1")
    if args.inner_folds != 0 and args.inner_folds < 2:
        parser.error("--inner-folds debe ser 0 o >=2")
    if args.bootstrap < 1 or args.max_sujetos < 0:
        parser.error("--bootstrap >=1 y --max-sujetos >=0")
    if args.csv is None:
        if args.modo == "evaluar":
            args.csv = previous_output / f"caracteristicas_5ML_GSP_{args.ventana:g}s.csv"
        else:
            args.csv = args.salida / f"caracteristicas_5ML_GSP_{args.ventana:g}s.csv"
    RUTA_BASE, ARCHIVO_LABELS, CARPETA_SALIDA = args.ruta,args.labels,args.salida
    VENTANA_SEGUNDOS, MIN_EPOCAS_POR_REGISTRO=args.ventana,args.min_epocas
    CARPETA_SALIDA.mkdir(parents=True,exist_ok=True)
    if args.modo in ("extraer","todo"):
        df, states = extraer_dataset_como_antes()
        states.to_csv(CARPETA_SALIDA/"estados_extraccion.csv",index=False)
        args.csv.parent.mkdir(parents=True,exist_ok=True)
        df.to_csv(args.csv,index=False)
        print("Características extraídas:",args.csv)
    if args.modo in ("evaluar","todo"):
        if not args.csv.exists():
            raise FileNotFoundError(
                f"No se encontró el CSV: {args.csv}. Indícalo mediante "
                "--csv \"RUTA\\caracteristicas_5ML_GSP_5s.csv\". No es necesario extraer nuevamente EEG."
            )
        df = pd.read_csv(args.csv)
        evaluar_xgboost(df,args.csv,CARPETA_SALIDA,args)
    print("Proceso terminado. Salida:",CARPETA_SALIDA)


if __name__ == "__main__":
    main()
