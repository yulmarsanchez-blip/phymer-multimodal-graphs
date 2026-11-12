# -*- coding: utf-8 -*-
"""COMPARACIÓN DE CINCO CLASIFICADORES + CARACTERÍSTICAS GSP — PHYMER.

Arquitectura: EEG .set -> ICA 1-45 Hz ajustada en copia -> ICA aplicada al raw
original -> CAR -> épocas 5s -> wPLI multitaper + potencia multitaper ->
grafos + GSP -> comparación ML (SVM, XGBoost, ExtraTrees, ElasticNet, LDA).

Cada registro produce UNA FILA, no una fila por época. La PSD y el grafo
se calculan con los mismos datos del registro. LOSO externo deja sujeto
completo fuera. CV interno agrupado evita filtraciones de etiquetas; ajuste
imputador, podas, estandarización y SelectKBest SOLO en entrenamiento.

Comparación primaria: sin_gsp (aristas+grafo+potencia) vs full (idéntico+GSP).
La topología normalizada adicional (lambda2, entropía) queda dentro de
'grafo' en ambas ramas: así el delta aísla únicamente las nuevas medidas
GSP con señal nodal.

IMPORTANTE: los resultados comparativos son exploratorios; si se escoge
el mejor modelo mirando LOSO externo, esa selección añade optimismo.
Comparar también contra mayoría y ejecutar confirmación externa antes
 de afirmar superioridad definitiva.
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
from sklearn.svm import SVC


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
SEARCH_N_JOBS = 1
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
# 7. VALIDACIÓN LOSO ANIDADA, CINCO ALGORITMOS, ABLACIÓN GSP
# =====================================================================

import argparse
import os
import platform
import sys
from functools import partial
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, classification_report, f1_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline


class CorrelationPruner(BaseEstimator, TransformerMixin):
    """Poda |Pearson|>threshold; ajuste SOLO con X de entrenamiento.

    Es un filtro no supervisado después de imputación y varianza.
    Se conserva la primera columna en orden determinista.
    """
    def __init__(self, threshold=0.995):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X debe ser bidimensional")
        c = np.corrcoef(X, rowvar=False)
        c = np.nan_to_num(c, nan=0.0)
        keep = np.ones(X.shape[1], bool)
        for i in range(X.shape[1]):
            if keep[i]:
                correlated = (np.abs(c[i, i+1:]) > self.threshold)
                keep[i+1:] &= ~correlated
        self.support_ = keep
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self):
        return self.support_.copy()


class SafeSelectKBest(BaseEstimator, TransformerMixin):
    """SelectKBest limitado al número de columnas supervivientes del fold.

    Evita que k=120 falle cuando un subconjunto solo tiene 56 variables
    o cuando la poda por correlación elimina más columnas de lo previsto.
    """
    def __init__(self, score_func=f_classif, k="all"):
        self.score_func = score_func
        self.k = k

    def fit(self, X, y):
        effective = min(self.k, np.asarray(X).shape[1]) if isinstance(self.k, int) else self.k
        self.selector_ = SelectKBest(score_func=self.score_func, k=effective)
        self.selector_.fit(X, y)
        return self

    def transform(self, X):
        return self.selector_.transform(X)

    def get_support(self):
        return self.selector_.get_support()


class BalancedXGBClassifier(ClassifierMixin, BaseEstimator):
    """XGB multiclase: codifica las clases locales y pondera por frecuencia
    exclusivamente en .fit(), evitando usar prevalencias del fold interno de validación.
    """
    def __init__(self, max_depth=2, n_estimators=200, learning_rate=0.05,
                 min_child_weight=3.0, subsample=0.8, colsample_bytree=0.8,
                 reg_lambda=3.0, random_state=2026):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state

    def fit(self, X, y):
        try:
            from xgboost import XGBClassifier
        except ImportError as ex:
            raise ImportError("Necesitas instalar xgboost: python -m pip install xgboost") from ex
        self.classes_, local = np.unique(np.asarray(y), return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("XGBoost necesita al menos dos clases")
        freq = np.bincount(local)
        weights = len(local) / (len(freq) * freq[local])
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, min_child_weight=self.min_child_weight,
            subsample=self.subsample, colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda, objective="multi:softprob",
            num_class=len(self.classes_), eval_metric="mlogloss", tree_method="hist",
            n_jobs=1, random_state=self.random_state,
        )
        self.model_.fit(X, local, sample_weight=weights)
        return self

    def predict(self, X):
        return self.classes_[self.model_.predict(X).astype(int)]

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def modelo_y_grid(nombre, selector="anova", escalador="standard", n_jobs_model=1):
    if selector == "anova":
        score_func = f_classif
    elif selector == "mi":
        score_func = partial(mutual_info_classif, random_state=2026)
    else:
        raise ValueError("selector debe ser anova o mi")

    if nombre == "svm_rbf":
        clf = SVC(kernel="rbf", class_weight="balanced", random_state=2026)
        grid = {"select__k": [120, "all"], "clf__C": [0.1, 1., 10.]}
    elif nombre == "xgboost":
        clf = BalancedXGBClassifier()
        grid = {"select__k": [120, "all"], "clf__max_depth": [2, 3],
                "clf__n_estimators": [150, 300]}
    elif nombre == "extra_trees":
        clf = ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                   n_jobs=n_jobs_model, random_state=2026)
        grid = {"select__k": [120, "all"], "clf__max_features": ["sqrt", 0.5],
                "clf__min_samples_leaf": [2, 5]}
    elif nombre == "elastic_net":
        clf = LogisticRegression(solver="saga", penalty="elasticnet", class_weight="balanced",
                                 max_iter=3500, tol=2e-3, random_state=2026)
        grid = {"select__k": [120, "all"], "clf__C": [0.3, 3.0],
                "clf__l1_ratio": [0.25, 0.75]}
    elif nombre == "lda_shrinkage":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        grid = {"select__k": [60, 120, "all"], "clf__shrinkage": ["auto", 0.5]}
    else:
        raise ValueError(f"Modelo desconocido: {nombre}")

    if escalador == "standard":
        sc = StandardScaler()
    elif escalador == "robust":
        sc = RobustScaler()
    else:
        raise ValueError("escalador: standard o robust")
    # Se conserva idéntico preprocesamiento para TODOS los clasificadores.
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold(threshold=1e-12)),
        ("pruner", CorrelationPruner(threshold=0.995)),
        ("scaler", sc),
        ("select", SafeSelectKBest(score_func=score_func, k="all")),
        ("clf", clf),
    ])
    return pipe, grid


def caracteristicas_por_grupo(df):
    meta = {"codigo", "sujeto", "video", "objetivo", "ventana_s", "n_epocas",
            "duracion_registro_s", "sfreq_hz", "n_componentes_ica_excluidos"}
    all_cols = sorted(c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c]))
    edge = [c for c in all_cols if c.startswith("edge_")]
    bandpower = [c for c in all_cols if c.startswith("bandpower_")]
    gsp = [c for c in all_cols if c.startswith("gsp_")]
    graph = [c for c in all_cols if c not in set(edge + bandpower + gsp)]
    return {
        "full": all_cols,
        "sin_gsp": sorted(edge + graph + bandpower),
        "grafo_aristas": sorted(edge + graph),
        "solo_grafo": graph,
        "solo_aristas": edge,
        "solo_potencia": bandpower,
        "grafo_potencia_gsp": sorted(graph + bandpower + gsp),
    }


def verificacion_dataset(df):
    required = {"sujeto", "codigo", "objetivo", "ventana_s", "n_epocas"}
    if not required.issubset(df.columns):
        raise ValueError(f"Faltan columnas obligatorias: {sorted(required-set(df.columns))}")
    if df["codigo"].duplicated().any():
        raise ValueError("Existen registros duplicados según codigo")
    if (df["ventana_s"] != VENTANA_SEGUNDOS).any():
        raise ValueError(f"Se necesita una tabla de características extraída con {VENTANA_SEGUNDOS}s; "
                         f"ventanas encontradas: {df['ventana_s'].unique()}")
    required_gsp = {f"gsp_{b}_dirichlet" for b in BAND_NAMES}
    if not required_gsp.issubset(df.columns):
        raise ValueError("Este CSV no contiene GSP; reextrae desde archivos .set. "
                         f"Faltan {required_gsp-set(df.columns)}")
    if df["sujeto"].nunique() < 3:
        raise ValueError("Se requieren al menos tres sujetos")
    if df["objetivo"].isna().any():
        raise ValueError("Hay etiquetas vacías")
    sets = caracteristicas_por_grupo(df)
    if not sets["solo_aristas"] or not sets["solo_grafo"] or not sets["solo_potencia"]:
        raise ValueError("Faltan familias de características en CSV")
    for c in sets["full"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if np.isinf(df[sets["full"]].to_numpy(dtype=float)).any():
        raise ValueError("Hay características infinitas")
    print("REGISTROS:", len(df), "SUJETOS:",df["sujeto"].nunique(),
          "CLASES:", df["objetivo"].nunique())
    for name in ["solo_aristas","solo_grafo","solo_potencia","full"]:
        print(" ",name,"variables:",len(sets[name]))
    if df["sujeto"].nunique() != 30:
        warnings.warn("No son 30 sujetos: prueba parcial o datos incompletos")
    return sets


def obtener_nombres_seleccionados(pipe, names):
    names = np.asarray(names)
    names = names[pipe.named_steps["variance"].get_support()]
    names = names[pipe.named_steps["pruner"].get_support()]
    return names[pipe.named_steps["select"].get_support()].tolist()


def crear_splits_internos(y, grupos, inner_folds=5):
    if inner_folds == 0:
        splitter = LeaveOneGroupOut()
    else:
        splitter = StratifiedGroupKFold(n_splits=min(inner_folds, len(np.unique(grupos))),
                                        shuffle=True, random_state=2026)
    dummy = np.zeros((len(y), 1))
    splits = list(splitter.split(dummy, y, grupos))
    if any(len(np.unique(y[tr])) < 2 for tr, va in splits):
        raise ValueError("Alguna partición interna no contiene dos clases de entrenamiento")
    return splits


def metrics(y, pred, classes):
    # BA fija = media del recall con las siete clases, incluso si alguna
    # no aparece en un fold (se le asigna 0). Evita cambiar denominadores.
    from sklearn.metrics import recall_score
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy_7clases": float(recall_score(y, pred, labels=classes,
                                               average="macro", zero_division=0)),
        "macro_f1_7clases": float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
    }


def evaluar_comparaciones(df, output, inner_folds=5, selector="anova", scaler="standard",
                           n_boot=1000, con_ablaciones=False, solo_modelos=None):
    groups = df["sujeto"].astype(int).to_numpy()
    enc = LabelEncoder()
    y = enc.fit_transform(df["objetivo"].astype(str).to_numpy())
    classes = np.arange(len(enc.classes_))
    subsets = verificacion_dataset(df)
    models = ["svm_rbf","xgboost","extra_trees","elastic_net","lda_shrinkage"]
    if solo_modelos:
        models = solo_modelos
    experiments = [(m, fs) for fs in ("sin_gsp", "full") for m in models]
    if con_ablaciones:
        experiments += [("svm_rbf", fs) for fs in
                        ("grafo_aristas", "solo_grafo", "solo_aristas",
                         "solo_potencia", "grafo_potencia_gsp")]
    outer_splits = list(LeaveOneGroupOut().split(df, y, groups))
    preds = []
    folds = []
    selected = []
    innerlog = []
    baseline_done = False
    for model_name, feature_set in experiments:
        cols = subsets[feature_set]
        X = df[cols].to_numpy(dtype=float)
        pipe, param_grid = modelo_y_grid(model_name, selector, scaler)
        print("\n"+"="*65)
        print("Modelo:",model_name,"| Variables:",feature_set,len(cols),
              "| Combinaciones:", int(np.prod([len(x) for x in param_grid.values()])))
        for fold, (tr, te) in enumerate(outer_splits, start=1):
            inner = crear_splits_internos(y[tr], groups[tr], inner_folds)
            # Scorer con vocabulario fijo. Global vocabulary es metadato, no se ajusta con y_test.
            def score(estimator, xx, yy):
                return f1_score(yy, estimator.predict(xx), labels=classes,
                                average="macro", zero_division=0)
            search = GridSearchCV(pipe, param_grid, scoring=score, cv=inner,
                                  n_jobs=3, refit=True, error_score="raise", return_train_score=False, pre_dispatch=3)
            search.fit(X[tr], y[tr])
            yhat = search.best_estimator_.predict(X[te])
            selected_names = obtener_nombres_seleccionados(search.best_estimator_, cols)
            mr = metrics(y[te], yhat, classes)
            frow = dict(modelo=model_name, feature_set=feature_set, fold=fold,
                        sujeto=int(groups[te[0]]), n_registros=len(te),
                        n_train=len(tr), n_feats_input=len(cols),
                        n_feats_selected=len(selected_names),
                        cv_macro_f1_interno=float(search.best_score_),
                        best_params=json.dumps(search.best_params_, sort_keys=True),
                        **mr)
            folds.append(frow)
            for name in selected_names:
                selected.append({"modelo":model_name,"feature_set":feature_set,
                                 "fold":fold,"caracteristica":name})
            for k, gi in enumerate(te):
                preds.append({"modelo":model_name, "feature_set":feature_set,
                              "sujeto":int(groups[gi]), "codigo":str(df.iloc[gi]["codigo"]),
                              "objetivo_real":str(enc.classes_[y[gi]]),
                              "objetivo_predicho":str(enc.classes_[yhat[k]]),
                              "real_code":int(y[gi]),"pred_code":int(yhat[k])})
            innerlog.extend([{"modelo":model_name,"feature_set":feature_set,
                              "fold":fold,"params":str(search.cv_results_["params"][j]),
                              "mean_inner_f1":float(search.cv_results_["mean_test_score"][j]),
                              "rank":int(search.cv_results_["rank_test_score"][j])}
                             for j in range(len(search.cv_results_["params"]))])
            print(f"  {fold:02d}/{len(outer_splits)} sujeto {int(groups[te[0]]):02d}"
                  f" F1={mr['macro_f1_7clases']:.3f}"
                  f" BA={mr['balanced_accuracy_7clases']:.3f}"
                  f" interno={search.best_score_:.3f}",flush=True)

        # Checkpoints por experimento (si la búsqueda se interrumpe).
        pd.DataFrame(preds).to_csv(output/"CHECKPOINT_predicciones.csv",index=False)
        pd.DataFrame(folds).to_csv(output/"CHECKPOINT_metricas.csv",index=False)
        pd.DataFrame(selected).to_csv(output/"CHECKPOINT_seleccion.csv",index=False)
    p = pd.DataFrame(preds)
    f = pd.DataFrame(folds)
    s = pd.DataFrame(selected)
    p.to_csv(output/"predicciones_modelos_LOSO.csv", index=False)
    f.to_csv(output/"metricas_por_sujeto_modelos_LOSO.csv", index=False)
    s.to_csv(output/"caracteristicas_seleccionadas_modelos.csv", index=False)
    pd.DataFrame(innerlog).to_csv(output/"ajuste_interno_cv.csv", index=False)
    summ=[]
    for (model_name, fs), q in p.groupby(["modelo","feature_set"], sort=False):
        m=metrics(q.real_code.to_numpy(),q.pred_code.to_numpy(),classes)
        sf=f[(f.modelo==model_name)&(f.feature_set==fs)]
        summ.append(dict(modelo=model_name,feature_set=fs,n_registros=len(q),
                         n_sujetos=q.sujeto.nunique(),
                         fold_macro_f1_mean=sf.macro_f1_7clases.mean(),
                         fold_macro_f1_sd=sf.macro_f1_7clases.std(ddof=1),
                         fold_BA_mean=sf.balanced_accuracy_7clases.mean(),**m))
        cm=confusion_matrix(q.real_code,q.pred_code,labels=classes)
        pd.DataFrame(cm,index=["real_"+c for c in enc.classes_],
                     columns=["pred_"+c for c in enc.classes_]).to_csv(
                     output/f"confusion_{model_name}_{fs}.csv")
    summary = pd.DataFrame(summ)
    summary.to_csv(output/"resumen_comparacion_modelos.csv", index=False)

    # Referencia pareada sin GSP por algoritmo y baseline mayoritario.
    base = {}
    for fold,(tr,te) in enumerate(outer_splits,1):
        dum=DummyClassifier(strategy="most_frequent").fit(np.zeros((len(tr),1)),y[tr])
        for index,pred in zip(te,dum.predict(np.zeros((len(te),1)))):
            base[str(df.iloc[index]["codigo"])]=int(pred)
    baseline_df=df[["codigo","sujeto"]].copy()
    baseline_df["real_code"]=y
    baseline_df["pred_code"]=baseline_df.codigo.map(base).astype(int)
    baseline_df["objetivo_real"]=[enc.classes_[v] for v in y]
    baseline_df["objetivo_predicho"]=[enc.classes_[v] for v in baseline_df.pred_code]
    baseline_df.to_csv(output/"baseline_LOSO.csv",index=False)

    # Bootstrap pareado: remuestrear la MISMA secuencia de sujetos para
    # dos modelos, recomputar macro-F1 pooled y diferencia en cada réplica.
    q_by = {(m,fs): t.set_index("codigo") for (m,fs),t in p.groupby(["modelo","feature_set"])}
    subject_ids=np.unique(groups)
    codes_by_subj={sid:df.loc[df.sujeto==sid,"codigo"].tolist() for sid in subject_ids}
    seed=np.random.default_rng(2026)
    sampled_subjects = seed.choice(subject_ids,size=(n_boot,len(subject_ids)),replace=True)
    ci=[]
    for (model_name,fs),q in q_by.items():
        comparator = q_by.get((model_name,"sin_gsp")) if fs=="full" else None
        cand_scores=[]; deltas=[]; deltas_ba=[]
        for sampled in sampled_subjects:
            codes=[code for sid in sampled for code in codes_by_subj[sid]]
            true=q.loc[codes,"real_code"].to_numpy()
            pred=q.loc[codes,"pred_code"].to_numpy()
            score=f1_score(true,pred,labels=classes,average="macro",zero_division=0)
            cand_scores.append(score)
            if comparator is not None:
                pred0=comparator.loc[codes,"pred_code"].to_numpy()
                deltas.append(score-f1_score(true,pred0,labels=classes,
                                            average="macro",zero_division=0))
                from sklearn.metrics import recall_score
                deltas_ba.append(recall_score(true,pred,labels=classes,average="macro",zero_division=0)
                                 -recall_score(true,pred0,labels=classes,average="macro",zero_division=0))
        ci.append({"modelo":model_name,"feature_set":fs,
                   "bootstrap_f1_ci_low":float(np.quantile(cand_scores,0.025)),
                   "bootstrap_f1_ci_high":float(np.quantile(cand_scores,0.975)),
                   "delta_f1_GSP_vs_sin_GSP":(
                       float(f1_score(q.real_code,q.pred_code,labels=classes,average="macro",zero_division=0)-
                             f1_score(comparator.loc[q.index,"real_code"],comparator.loc[q.index,"pred_code"],
                                      labels=classes,average="macro",zero_division=0))
                       if comparator is not None else np.nan),
                   "delta_f1_ci_low":float(np.quantile(deltas,0.025)) if deltas else np.nan,
                   "delta_f1_ci_high":float(np.quantile(deltas,0.975)) if deltas else np.nan,
                   "delta_ba_ci_low":float(np.quantile(deltas_ba,0.025)) if deltas_ba else np.nan,
                   "delta_ba_ci_high":float(np.quantile(deltas_ba,0.975)) if deltas_ba else np.nan})
    cidf=pd.DataFrame(ci)
    cidf.to_csv(output/"bootstrap_pareado_por_sujeto.csv",index=False)

    # Comparaciones pareadas adicionales entre los cinco clasificadores
    # en el MISMO conjunto de variables FULL, frente a SVM y Dummy.
    paired_rows = []
    reference_key = ("svm_rbf", "full")
    svm_reference = q_by.get(reference_key)
    dummy = baseline_df.set_index("codigo")
    for (model_name, fs), q in q_by.items():
        if fs != "full":
            continue
        for ref_name, reference in (("svm_rbf", svm_reference), ("baseline_mayoritaria", dummy)):
            if reference is None or (model_name == "svm_rbf" and ref_name == "svm_rbf"):
                continue
            raw_true=q.real_code.to_numpy()
            point_delta=(f1_score(raw_true,q.pred_code.to_numpy(),labels=classes,
                                  average="macro",zero_division=0) -
                         f1_score(raw_true,reference.loc[q.index,"pred_code"].to_numpy(),
                                  labels=classes,average="macro",zero_division=0))
            replicate_deltas=[]
            for sampled in sampled_subjects:
                codes=[code for sid in sampled for code in codes_by_subj[sid]]
                true=q.loc[codes,"real_code"].to_numpy()
                candidate_pred=q.loc[codes,"pred_code"].to_numpy()
                reference_pred=reference.loc[codes,"pred_code"].to_numpy()
                replicate_deltas.append(f1_score(true,candidate_pred,labels=classes,
                                    average="macro",zero_division=0)-
                              f1_score(true,reference_pred,labels=classes,
                                    average="macro",zero_division=0))
            paired_rows.append({"modelo":model_name,"feature_set":fs,
                         "referencia":ref_name,"delta_macro_f1_puntual":float(point_delta),
                         "delta_ic_low":float(np.quantile(replicate_deltas,0.025)),
                         "delta_ic_high":float(np.quantile(replicate_deltas,0.975))})
    pd.DataFrame(paired_rows).to_csv(output/"comparaciones_pareadas_modelos_full.csv",index=False)
    print("\nRESUMEN\n",summary.to_string(index=False))
    print("\nIC PAREADOS\n",cidf.to_string(index=False))
    return summary


# =====================================================================
# 8. DATOS: EXTRACCIÓN EN LOS 30 SUJETOS, PERSISTENCIA Y OPCIÓN REUSO
# =====================================================================

def extraer_dataset_ampliado():
    if not RUTA_BASE.exists():
        raise FileNotFoundError(f"No existe carpeta Emotiv: {RUTA_BASE}")
    labels = cargar_labels()
    rows, states = [], []
    tasks = [(s,v) for s in SUJETOS for v in VIDEOS]
    for pos, (subject, video) in enumerate(tasks,1):
        code = f"SUB{subject:02d}VID{video:02d}"
        ruta = localizar_archivo(subject,video)
        if ruta is None:
            states.append({"codigo":code,"estado":"SIN_ARCHIVO"});continue
        if code not in labels.index:
            states.append({"codigo":code,"estado":"SIN_ETIQUETA"});continue
        target=labels.loc[code,TARGET_COLUMN]
        if pd.isna(target) or str(target).strip()=="":
            states.append({"codigo":code,"estado":"ETIQUETA_VACIA"});continue
        print(f"EXTRACCIÓN {pos}/{len(tasks)} {code}",flush=True)
        try:
            raw, excluded=preparar_eeg(ruta)
            sfreq=float(raw.info["sfreq"])
            dur=float(raw.times[-1])
            epochs=mne.make_fixed_length_epochs(raw,duration=VENTANA_SEGUNDOS,
                     overlap=0.,preload=True,reject_by_annotation=True,verbose=False)
            if len(epochs)<MIN_EPOCAS_POR_REGISTRO:
                states.append({"codigo":code,"estado":"EPOCAS_INSUFICIENTES",
                               "n_epocas":len(epochs)});continue
            matrices=calcular_matrices_wpli(epochs)
            powers=potencia_nodal_por_banda(epochs)
            features=extraer_caracteristicas_ampliadas(matrices,powers)
            rows.append(dict(codigo=code,sujeto=subject,video=video,
                             objetivo=str(target),ventana_s=VENTANA_SEGUNDOS,
                             n_epocas=len(epochs),duracion_registro_s=dur,
                             sfreq_hz=sfreq,n_componentes_ica_excluidos=len(excluded),
                             **features))
            states.append(dict(codigo=code,estado="OK",n_epocas=len(epochs),
                               componentes_ica_excluidos=str(excluded),
                               aviso_epocas_escasas=bool(len(epochs)<8)))
        except Exception as ex:
            states.append({"codigo":code,"estado":f"ERROR {type(ex).__name__}: {ex}"})
            print("ERROR",code,type(ex).__name__,ex,flush=True)
    if not rows:
        raise RuntimeError("No se extrajeron registros válidos. Revisar estados.")
    return pd.DataFrame(rows),pd.DataFrame(states)


def audit_metrics(df,out):
    # Controles adicionales para no confundir fallo lateralidad con interpretación.
    if not set(f"edge_Beta_{i}_{j}" for i,j in [("AF3","F7")]).issubset(df.columns):
        return
    left={"AF3","F7","F3","FC5","T7","P7","O1"}
    right=set(CANALES)-left
    rows=[]
    for _,r in df.iterrows():
        rr={"codigo":r.codigo,"sujeto":r.sujeto,"objetivo":r.objetivo}
        for band in BAND_NAMES:
            vals={"izq":[],"der":[],"entre":[]}
            for i,j in __import__('itertools').combinations(CANALES,2):
                col=f"edge_{band}_{i}_{j}"
                if col not in r.index:continue
                key="izq" if i in left and j in left else "der" if i in right and j in right else "entre"
                vals[key].append(float(r[col]))
            for key,v in vals.items():
                rr[f"wpli_{band}_{key}"]=float(np.mean(v)) if v else np.nan
        rows.append(rr)
    pd.DataFrame(rows).to_csv(out/"control_lateralidad_wpli.csv",index=False)


def ajustar_modelo_con_todos(df, out, model_name, feature_set, selector, scaler, inner_folds):
    """Modelo de DESPLIEGUE: no produce una evaluación externa independiente.

    La selección del modelo/familia por mirar los scores externos introduce
    sesgo de selección; validar la elección en nuevos sujetos antes de
    afirmar que es la mejor arquitectura.
    """
    subsets=verificacion_dataset(df)
    if feature_set not in subsets:
        raise ValueError(f"Feature set desconocido: {feature_set}")
    names=subsets[feature_set]
    X=df[names].to_numpy(dtype=float)
    encoder=LabelEncoder()
    y=encoder.fit_transform(df.objetivo.astype(str).to_numpy())
    groups=df.sujeto.astype(int).to_numpy()
    splits=crear_splits_internos(y,groups,inner_folds)
    pipe,grid=modelo_y_grid(model_name,selector,scaler)
    def score(estimator,xx,yy):
        return f1_score(yy,estimator.predict(xx),labels=np.arange(len(encoder.classes_)),
                        average="macro",zero_division=0)
    search=GridSearchCV(pipe,grid,scoring=score,cv=splits,n_jobs=3,
                        refit=True,error_score="raise")
    search.fit(X,y)
    path=out/f"modelo_despliegue_{model_name}_{feature_set}.joblib"
    joblib.dump({"model":search.best_estimator_,"features":names,
                 "classes":encoder.classes_.tolist(),
                 "preprocessing": "ICA 1-45Hz copy, applied raw, CAR, no crop",
                 "wpli": "multitaper bandwidth=2Hz, faverage four bands",
                 "window_seconds":VENTANA_SEGUNDOS,
                 "best_params":search.best_params_},path)
    print("Modelo entrenado con todos los sujetos:",path)
    print("CV interno tras selección:",search.best_score_,"— NO es rendimiento externo.")


def version_info(out,args):
    import sklearn
    manifest={"python":sys.version, "platform":platform.platform(),
              "numpy":np.__version__, "pandas":pd.__version__,
              "sklearn":sklearn.__version__,"mne":mne.__version__,
              "networkx":nx.__version__,
              "window_seconds":VENTANA_SEGUNDOS,
              "min_epochs":MIN_EPOCAS_POR_REGISTRO,
              "ica_reference":"CAR", "ica_filter":[ICA_L_FREQ,ICA_H_FREQ],
              "ica_eye_proxies":["AF3","AF4"], "wpli_mt_bandwidth":MT_BANDWIDTH_HZ,
              "bands":BANDAS,"selectors":args.selector,
              "scaler":args.scaler,"inner_folds":args.inner_folds,
              "random_state":2026}
    try:
        import xgboost;manifest["xgboost"]=xgboost.__version__
    except ImportError:pass
    try:
        import mne_connectivity;manifest["mne_connectivity"]=mne_connectivity.__version__
    except ImportError:pass
    (out/"configuracion_y_versiones.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")


def main():
    global RUTA_BASE,ARCHIVO_LABELS,CARPETA_SALIDA,VENTANA_SEGUNDOS,MIN_EPOCAS_POR_REGISTRO
    parser=argparse.ArgumentParser(description="Comparar 5 ML + GSP para EEG/PhyMER (nested LOSO)")
    parser.add_argument("--ruta",type=Path,default=RUTA_BASE,help="carpeta Emotiv con .set")
    parser.add_argument("--labels",type=Path,default=ARCHIVO_LABELS,help="labels.csv")
    parser.add_argument("--salida",type=Path,default=CARPETA_SALIDA)
    parser.add_argument("--ventana",type=float,choices=[3.,5.],default=5.)
    parser.add_argument("--min-epocas",type=int,default=5)
    parser.add_argument("--modo",choices=["todo","extraer","evaluar"],default="todo")
    parser.add_argument("--csv",type=Path,default=None,
                        help="Para modo evaluar: tabla nueva generada con este programa")
    parser.add_argument("--inner-folds",type=int,default=5,
                        help="5=StratifiedGroupKFold de 5 grupos; 0=LOSO interno exhaustivo")
    parser.add_argument("--selector",choices=["anova","mi"],default="anova")
    parser.add_argument("--scaler",choices=["standard","robust"],default="standard")
    parser.add_argument("--bootstrap",type=int,default=1000)
    parser.add_argument("--ablaciones",action="store_true",
                        help="Añade SVM para subsets secundarios")
    parser.add_argument("--modelos",nargs="+",choices=["svm_rbf","xgboost","extra_trees",
              "elastic_net","lda_shrinkage"],default=None,help="Opcional: prueba uno o varios")
    parser.add_argument("--ajustar-final",choices=["svm_rbf","xgboost","extra_trees",
           "elastic_net","lda_shrinkage"],default=None,
           help="Opcional: reajustar modelo elegido con todos los sujetos y guardar joblib")
    parser.add_argument("--ajustar-feature-set",choices=["full","sin_gsp","grafo_aristas",
           "solo_grafo","solo_aristas","solo_potencia","grafo_potencia_gsp"],
           default="full")
    args=parser.parse_args()
    RUTA_BASE,ARCHIVO_LABELS,CARPETA_SALIDA=args.ruta,args.labels,args.salida
    VENTANA_SEGUNDOS=args.ventana
    MIN_EPOCAS_POR_REGISTRO=args.min_epocas
    CARPETA_SALIDA.mkdir(parents=True,exist_ok=True)
    version_info(CARPETA_SALIDA,args)
    data_path=args.csv or (CARPETA_SALIDA/f"caracteristicas_5ML_GSP_{VENTANA_SEGUNDOS:g}s.csv")
    if args.modo in ("todo","extraer"):
        df,state=extraer_dataset_ampliado()
        state.to_csv(CARPETA_SALIDA/"estados_extraccion.csv",index=False)
        df.to_csv(data_path,index=False)
        audit_metrics(df,CARPETA_SALIDA)
        print("Características guardadas:",data_path)
    if args.modo in ("todo","evaluar"):
        if not data_path.exists():
            raise FileNotFoundError(f"Falta dataset calculado: {data_path}")
        df=pd.read_csv(data_path)
        audit_metrics(df,CARPETA_SALIDA)
        evaluar_comparaciones(df,CARPETA_SALIDA,inner_folds=args.inner_folds,
                              selector=args.selector,scaler=args.scaler,
                              n_boot=args.bootstrap,con_ablaciones=args.ablaciones,
                              solo_modelos=args.modelos)
        if args.ajustar_final is not None:
            ajustar_modelo_con_todos(df,CARPETA_SALIDA,args.ajustar_final,
                                    args.ajustar_feature_set,args.selector,args.scaler,
                                    args.inner_folds)
    print("Proceso terminado. Carpeta:",CARPETA_SALIDA)


if __name__=="__main__":
    main()
