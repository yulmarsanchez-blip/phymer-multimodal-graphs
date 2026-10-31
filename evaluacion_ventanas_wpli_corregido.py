# -*- coding: utf-8 -*-
"""
Evaluación de duraciones de época para conectividad EEG mediante wPLI.

Compara ventanas de 1, 2, 3, 4 y 5 s (o cualquier lista definida por el
usuario) mediante:
  1) Adecuación espectral: ciclos a 4 Hz, resolución nominal 1/T y número
     de épocas disponibles.
  2) Estabilidad: split-half repetido de épocas y comparación entre matrices
     wPLI (correlación de las 91 aristas, RMSE y MAE).
  3) Utilidad en clasificación: Leave-One-Subject-Out (LOSO), evaluada con
     SVM y dos representaciones de características: pesos de aristas y
     métricas de grafo.

El preprocesamiento replica la estructura del script anterior:
  - leer EEGLAB .set
  - seleccionar EEG
  - ajustar ICA en una copia filtrada 1-45 Hz
  - detectar componentes oculares con AF3/AF4 como proxies
  - aplicar ICA a la señal original sin el filtro 1-45 Hz
  - re-referenciar a promedio
  - NO recortar automáticamente los extremos

IMPORTANTE:
- Para una comparación justa, mt_bandwidth se mantiene fijo entre duraciones.
- Los registros se consideran muestras (sujeto, video); LOSO retiene sujetos
  completos fuera del entrenamiento.
- Con un solo sujeto, LOSO no se calcula: el script lo informa explícitamente.
- Los criterios espectrales son descriptores orientativos, no reglas físicas
  universales de aceptación/rechazo.
"""

from __future__ import annotations

import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne
import networkx as nx

from mne_connectivity import spectral_connectivity_epochs
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# =====================================================================
# 1. CONFIGURACIÓN EDITABLE
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
    r"C:\Users\Maria\Downloads\Code-TG\Analisis-Ventana"
) / "evaluacion_ventanas"

# Cambiar manualmente para cada iteración. Ejemplos:
# VENTANAS_SEGUNDOS = [2.0, 3.0, 4.0]
# VENTANAS_SEGUNDOS = [1.0, 2.0, 3.0, 4.0, 5.0]
VENTANAS_SEGUNDOS = [1.0, 2.0, 3.0, 4.0, 5.0]

# Para una prueba inicial con SUB01 y los videos revisados:
SUJETOS = list(range(1, 31))
VIDEOS = list(range(1, 26))

# Para todos los sujetos/carpetas detectados, usar:
# SUJETOS = None
# VIDEOS = None

# Variable objetivo para clasificación. Normalmente "emotion".
TARGET_COLUMN = "emotion"

# "edges": usa los 91 pesos únicos por banda (364 características).
# "graph": usa métricas de resumen de los cuatro grafos.
# Se pueden evaluar ambas en la misma ejecución.
MODOS_CARACTERISTICAS = ["edges", "graph"]

# Parámetros del preprocesamiento ICA (según el pipeline anterior).
ICA_L_FREQ = 1.0
ICA_H_FREQ = 45.0
ICA_RANDOM_STATE = 42
ICA_EOG_THRESHOLD = 3.0

# Ventanas de conectividad: se mantiene constante entre duraciones.
BANDAS = {
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 40.0),
}
MT_BANDWIDTH_HZ = 2.0
N_JOBS_WPLI = 1

# Estabilidad split-half. Aumentar N_SPLIT_REPEATS mejora la precisión de
# la estimación, pero aumenta el tiempo de ejecución.
CALCULAR_ESTABILIDAD = True
N_SPLIT_REPEATS = 3
MIN_EPOCAS_POR_MITAD = 5
RANDOM_STATE = 2026

# Mantener solo registros con al menos esta cantidad de épocas para la
# comparación de clasificación. El mismo conjunto de registros se utilizará
# para todas las duraciones cuando sea posible.
MIN_EPOCAS_CLASIFICACION = 8

# Mínimo de épocas antes de intentar el cálculo completo de wPLI.
MIN_EPOCAS_WPLI = 5

# Clasificador fijo para comparar ventanas; no se hace búsqueda de
# hiperparámetros para evitar que la complejidad de esta etapa oculte el
# efecto de la duración. La selección final de modelo se haría después.
SVC_C = 1.0
SVC_KERNEL = "rbf"
SVC_CLASS_WEIGHT = "balanced"

# Activar para la evaluación final. La evaluación LOSO por ventana es útil
# para exploración; si se selecciona la ventana ganadora con esos resultados,
# una evaluación anidada ofrece una estimación menos optimista.
CALCULAR_LOSO_ANIDADO = True


# =====================================================================
# 2. FUNCIONES DE ARCHIVOS Y ETIQUETAS
# =====================================================================

def extraer_codigo(path: Path) -> Optional[str]:
    """Extrae un código como SUB01VID01 del nombre del archivo."""
    coincidencia = re.search(r"(SUB\d+VID\d+)", path.stem, re.IGNORECASE)
    return coincidencia.group(1).upper() if coincidencia else None


def descubrir_sujetos() -> List[int]:
    """Devuelve los IDs SUB## presentes en RUTA_BASE."""
    encontrados = []
    if not RUTA_BASE.exists():
        return encontrados
    for carpeta in RUTA_BASE.iterdir():
        if carpeta.is_dir():
            m = re.fullmatch(r"SUB(\d+)", carpeta.name, re.IGNORECASE)
            if m:
                encontrados.append(int(m.group(1)))
    return sorted(set(encontrados))


def localizar_archivo(sujeto: int, video: int) -> Optional[Path]:
    """Busca el .set admitiendo nombres como SUB01VID01_eeg(1).set."""
    codigo = f"SUB{sujeto:02d}VID{video:02d}"
    candidatos = list(RUTA_BASE.rglob(f"{codigo}*_eeg*.set"))
    if not candidatos:
        # Alternativa tolerante por si el nombre no incluye exactamente _eeg.
        candidatos = list(RUTA_BASE.rglob(f"{codigo}*.set"))
        candidatos = [p for p in candidatos if extraer_codigo(p) == codigo]
    if not candidatos:
        return None

    # Preferir el nombre convencional y, en su defecto, un único candidato
    # determinista para evitar duplicar un mismo experimento.
    nombre_ideal = f"{codigo}_eeg.set".lower()
    candidatos = sorted(
        candidatos,
        key=lambda p: (
            p.name.lower() != nombre_ideal,
            len(p.name),
            p.name.lower(),
        ),
    )
    if len(candidatos) > 1:
        print(
            f"  AVISO: {codigo} tiene {len(candidatos)} archivos candidatos; "
            f"se utilizará: {candidatos[0].name}"
        )
    return candidatos[0]


def cargar_labels() -> pd.DataFrame:
    if not ARCHIVO_LABELS.exists():
        raise FileNotFoundError(f"No se encuentra labels.csv: {ARCHIVO_LABELS}")
    labels = pd.read_csv(ARCHIVO_LABELS)
    required = {"experiment_code", TARGET_COLUMN}
    faltantes = required - set(labels.columns)
    if faltantes:
        raise ValueError(
            f"Faltan columnas en labels.csv: {sorted(faltantes)}. "
            f"Columnas disponibles: {list(labels.columns)}"
        )
    labels = labels.copy()
    labels["experiment_code"] = (
        labels["experiment_code"].astype(str).str.strip().str.upper()
    )
    # Si hay códigos duplicados, no asignar etiquetas ambiguas en silencio.
    duplicados = labels.loc[
        labels["experiment_code"].duplicated(keep=False), "experiment_code"
    ].unique()
    if len(duplicados):
        raise ValueError(
            "Hay experiment_code duplicados en labels.csv; revisar antes de "
            f"continuar. Ejemplos: {duplicados[:5].tolist()}"
        )
    return labels.set_index("experiment_code", drop=False)


# =====================================================================
# 3. PREPROCESAMIENTO: MISMO CRITERIO QUE EL SCRIPT ANTERIOR
# =====================================================================

def preparar_eeg(ruta: Path) -> Tuple[mne.io.BaseRaw, List[int]]:
    """Preprocesa un .set y devuelve Raw limpio y componentes ICA excluidos."""
    raw = mne.io.read_raw_eeglab(str(ruta), preload=True, verbose=False)
    raw.pick("eeg")

    if len(raw.ch_names) < 4:
        raise ValueError(
            f"Muy pocos canales EEG tras raw.pick('eeg'): {raw.ch_names}"
        )

    # Ajustar ICA sobre una copia filtrada; el Raw que recibirá la ICA permanece
    # sin aplicar el FIR 1-45 Hz, igual que en el script original.
    raw_ica = raw.copy()
    raw_ica.filter(
        l_freq=ICA_L_FREQ,
        h_freq=ICA_H_FREQ,
        fir_design="firwin",
        verbose=False,
    )

    n_components = min(len(raw.ch_names), 14)
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=ICA_RANDOM_STATE,
        max_iter="auto",
    )
    ica.fit(raw_ica, verbose=False)

    componentes_af3 = []
    componentes_af4 = []
    if "AF3" in raw_ica.ch_names:
        try:
            componentes_af3, _ = ica.find_bads_eog(
                raw_ica,
                ch_name="AF3",
                threshold=ICA_EOG_THRESHOLD,
                verbose=False,
            )
        except Exception as exc:
            print(f"  AVISO: no se pudieron detectar proxies EOG en AF3: {exc}")
    if "AF4" in raw_ica.ch_names:
        try:
            componentes_af4, _ = ica.find_bads_eog(
                raw_ica,
                ch_name="AF4",
                threshold=ICA_EOG_THRESHOLD,
                verbose=False,
            )
        except Exception as exc:
            print(f"  AVISO: no se pudieron detectar proxies EOG en AF4: {exc}")

    componentes_eog = sorted(set(componentes_af3) | set(componentes_af4))
    ica.exclude = componentes_eog

    raw_limpio = raw.copy()
    ica.apply(raw_limpio, verbose=False)
    raw_limpio.set_eeg_reference(
        ref_channels="average",
        projection=False,
        verbose=False,
    )

    return raw_limpio, componentes_eog


# =====================================================================
# 4. wPLI Y REPRESENTACIONES DEL GRAFO
# =====================================================================

BAND_NAMES = list(BANDAS.keys())
FMIN = [BANDAS[b][0] for b in BAND_NAMES]
FMAX = [BANDAS[b][1] for b in BAND_NAMES]


def symmetrize_wpli(matrix: np.ndarray) -> np.ndarray:
    """Normaliza la salida densa de MNE a una matriz simétrica.

    Algunas versiones/formas de salida contienen solamente un triángulo
    calculado y ceros en el otro; promediar en ese caso dividiría los pesos
    por dos. Se detecta el triángulo con más pesos no nulos y se refleja.
    Si ambos triángulos parecen completos, se promedian.
    """
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
    out = np.clip(out, 0.0, None)
    return out


def calcular_matrices_wpli(epochs: mne.Epochs) -> Dict[str, np.ndarray]:
    """Estima una matriz wPLI por banda a partir del conjunto de épocas."""
    con = spectral_connectivity_epochs(
        epochs,
        method="wpli",
        mode="multitaper",
        fmin=FMIN,
        fmax=FMAX,
        faverage=True,
        mt_bandwidth=MT_BANDWIDTH_HZ,
        mt_adaptive=False,
        mt_low_bias=True,
        n_jobs=N_JOBS_WPLI,
        verbose=False,
    )
    densa = con.get_data(output="dense")
    if densa.ndim != 3 or densa.shape[-1] != len(BAND_NAMES):
        raise ValueError(f"Forma inesperada de la matriz wPLI: {densa.shape}")
    return {
        band: symmetrize_wpli(densa[:, :, idx])
        for idx, band in enumerate(BAND_NAMES)
    }


def edge_vector(matrix: np.ndarray) -> np.ndarray:
    """Vector de las 91 aristas únicas de una matriz simétrica de 14 nodos."""
    i, j = np.triu_indices(matrix.shape[0], k=1)
    return matrix[i, j].astype(float)


def feature_edges(matrices: Dict[str, np.ndarray], channels: List[str]) -> Dict[str, float]:
    """Usa cada peso de arista único como predictor, sin umbralizar el grafo."""
    features: Dict[str, float] = {}
    for band, matrix in matrices.items():
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                name = f"edge_{band}_{channels[i]}_{channels[j]}"
                features[name] = float(matrix[i, j])
    return features


def graph_summary_features(
    matrices: Dict[str, np.ndarray], channels: List[str]
) -> Dict[str, float]:
    """Métricas de resumen ponderadas por banda para una evaluación alternativa.

    No umbraliza la matriz: cada peso wPLI no negativo se conserva.
    """
    features: Dict[str, float] = {}
    n = len(channels)

    for band, matrix in matrices.items():
        A = np.asarray(matrix, dtype=float)
        i, j = np.triu_indices(n, k=1)
        edges = A[i, j]
        strengths = A.sum(axis=1)

        features[f"{band}_edge_mean"] = float(np.mean(edges))
        features[f"{band}_edge_std"] = float(np.std(edges))
        features[f"{band}_edge_median"] = float(np.median(edges))
        features[f"{band}_strength_mean"] = float(np.mean(strengths))
        features[f"{band}_strength_std"] = float(np.std(strengths))

        for name, value in zip(channels, strengths):
            features[f"{band}_strength_{name}"] = float(value)

        G = nx.Graph()
        G.add_nodes_from(channels)
        for idx_i in range(n):
            for idx_j in range(idx_i + 1, n):
                weight = float(A[idx_i, idx_j])
                if np.isfinite(weight) and weight > 0:
                    G.add_edge(
                        channels[idx_i],
                        channels[idx_j],
                        weight=weight,
                        distance=1.0 / max(weight, 1e-12),
                    )

        features[f"{band}_weighted_clustering"] = float(
            nx.average_clustering(G, weight="weight")
        )

        # Laplaciano combinatorio del grafo ponderado.
        L = np.diag(strengths) - A
        eigvals, eigvecs = np.linalg.eigh(L)
        eigvals = np.maximum(eigvals, 0.0)
        scale = max(float(np.max(eigvals)), 1.0)
        eig_tol = 1e-10 * scale
        positive = eigvals[eigvals > eig_tol]

        features[f"{band}_lambda2"] = float(eigvals[1]) if n >= 2 else np.nan
        features[f"{band}_fiedler_ipr"] = float(np.sum(eigvecs[:, 1] ** 4)) if n >= 2 else np.nan

        if len(positive) >= 2:
            p = positive / np.sum(positive)
            h = -np.sum(p * np.log(p + 1e-15))
            h_max = np.log(len(p))
            features[f"{band}_laplacian_entropy"] = float(h / h_max) if h_max > 0 else 0.0
        else:
            features[f"{band}_laplacian_entropy"] = 0.0

        # Kirchhoff index: definido para una red conectada; se deja NaN si
        # el grafo está desconectado por pesos exactamente nulos.
        if len(positive) == n - 1:
            features[f"{band}_kirchhoff"] = float(n * np.sum(1.0 / positive))
        else:
            features[f"{band}_kirchhoff"] = np.nan

        # Eficiencia global ponderada: la distancia de una arista es 1/wPLI.
        if G.number_of_edges() > 0 and nx.is_connected(G):
            lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="distance"))
            inverse_distances = []
            for ii in range(n):
                for jj in range(ii + 1, n):
                    d = lengths[channels[ii]][channels[jj]]
                    inverse_distances.append(1.0 / d if d > 0 else 0.0)
            features[f"{band}_global_efficiency_weighted"] = float(
                np.mean(inverse_distances)
            )
        else:
            features[f"{band}_global_efficiency_weighted"] = 0.0

    return features


# =====================================================================
# 5. ESTABILIDAD: SPLIT-HALF REPETIDO
# =====================================================================

def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(pearsonr(a, b)[0])


def calcular_estabilidad_split_half(
    epochs: mne.Epochs,
    codigo: str,
    window_s: float,
    subject: int,
    video: int,
) -> List[dict]:
    """Compara matrices wPLI estimadas en dos mitades disjuntas de épocas.

    Repite particiones aleatorias para reducir dependencia de una única
    división. La puntuación es reproducibilidad interna del registro, no una
    prueba de ausencia de artefactos ni una garantía de validez fisiológica.
    """
    n_epochs = len(epochs)
    min_total = 2 * MIN_EPOCAS_POR_MITAD
    if n_epochs < min_total:
        return [{
            "codigo": codigo,
            "sujeto": subject,
            "video": video,
            "ventana_s": window_s,
            "repeticion": np.nan,
            "banda": band,
            "n_epocas": n_epochs,
            "n_epocas_A": np.nan,
            "n_epocas_B": np.nan,
            "correlacion_aristas": np.nan,
            "rmse_aristas": np.nan,
            "mae_aristas": np.nan,
            "nrmse_aristas": np.nan,
            "estado": "INSUFICIENTES_EPOCAS_PARA_SPLIT_HALF",
        } for band in BAND_NAMES]

    seed = RANDOM_STATE + subject * 100_000 + video * 1_000 + int(round(window_s * 10))
    rng = np.random.default_rng(seed)
    indices = np.arange(n_epochs)
    n_a = n_epochs // 2
    rows: List[dict] = []

    for rep in range(N_SPLIT_REPEATS):
        perm = rng.permutation(indices)
        idx_a = perm[:n_a]
        idx_b = perm[n_a:]
        ep_a = epochs[idx_a]
        ep_b = epochs[idx_b]

        try:
            mat_a = calcular_matrices_wpli(ep_a)
            mat_b = calcular_matrices_wpli(ep_b)
        except Exception as exc:
            print(f"  AVISO estabilidad {codigo}, T={window_s}s: {exc}")
            for band in BAND_NAMES:
                rows.append({
                    "codigo": codigo,
                    "sujeto": subject,
                    "video": video,
                    "ventana_s": window_s,
                    "repeticion": rep + 1,
                    "banda": band,
                    "n_epocas": n_epochs,
                    "n_epocas_A": len(idx_a),
                    "n_epocas_B": len(idx_b),
                    "correlacion_aristas": np.nan,
                    "rmse_aristas": np.nan,
                    "mae_aristas": np.nan,
                    "nrmse_aristas": np.nan,
                    "estado": f"ERROR: {exc}",
                })
            continue

        for band in BAND_NAMES:
            ea = edge_vector(mat_a[band])
            eb = edge_vector(mat_b[band])
            delta = ea - eb
            rmse = float(np.sqrt(np.mean(delta ** 2)))
            mae = float(np.mean(np.abs(delta)))
            scale = float(np.std(np.concatenate([ea, eb])))
            nrmse = rmse / scale if scale > 1e-12 else np.nan
            corr = safe_correlation(ea, eb)

            rows.append({
                "codigo": codigo,
                "sujeto": subject,
                "video": video,
                "ventana_s": window_s,
                "repeticion": rep + 1,
                "banda": band,
                "n_epocas": n_epochs,
                "n_epocas_A": len(idx_a),
                "n_epocas_B": len(idx_b),
                "correlacion_aristas": corr,
                "rmse_aristas": rmse,
                "mae_aristas": mae,
                "nrmse_aristas": nrmse,
                "estado": "OK",
            })

    return rows


# =====================================================================
# 6. ADECUACIÓN ESPECTRAL Y NÚMERO DE ÉPOCAS
# =====================================================================

def registro_adecuacion(
    codigo: str,
    subject: int,
    video: int,
    duration_s: float,
    window_s: float,
    sfreq: float,
    n_epochs: int,
) -> dict:
    f_low = min(v[0] for v in BANDAS.values())
    cycles = f_low * window_s
    return {
        "codigo": codigo,
        "sujeto": subject,
        "video": video,
        "duracion_registro_s": duration_s,
        "ventana_s": window_s,
        "sfreq_hz": sfreq,
        "n_epocas": n_epochs,
        "ciclos_a_fmin_minimo": cycles,
        "resolucion_nominal_1_sobre_T_hz": 1.0 / window_s,
        "mt_bandwidth_hz_fijo": MT_BANDWIDTH_HZ,
        "producto_tiempo_ancho_banda_TB": window_s * MT_BANDWIDTH_HZ / 2.0,
        "al_menos_5_ciclos_a_4Hz": cycles >= 5.0,
        "n_epocas_suficientes_clasificacion": n_epochs >= MIN_EPOCAS_CLASIFICACION,
    }


# =====================================================================
# 7. CLASIFICACIÓN LOSO
# =====================================================================

def crear_clasificador() -> Pipeline:
    """Pipeline fijo; el escalado y la imputación se ajustan en cada fold."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svc", SVC(
            C=SVC_C,
            kernel=SVC_KERNEL,
            class_weight=SVC_CLASS_WEIGHT,
        )),
    ])


def columnas_de_caracteristicas(df: pd.DataFrame, feature_mode: str) -> List[str]:
    meta_cols = {
        "codigo", "sujeto", "video", "ventana_s", "n_epocas", "objetivo",
        "modo_caracteristicas",
    }
    if feature_mode == "edges":
        return [c for c in df.columns if c.startswith("edge_")]
    if feature_mode == "graph":
        return [c for c in df.columns if c not in meta_cols and not c.startswith("edge_")]
    raise ValueError(f"Modo de características no válido: {feature_mode}")


def evaluar_loso(
    df_features: pd.DataFrame,
    window_s: float,
    feature_mode: str,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evalúa LOSO sobre las mismas grabaciones disponibles para todas las T."""
    subset = df_features[df_features["ventana_s"] == window_s].copy()
    if subset.empty:
        return ({
            "ventana_s": window_s,
            "modo_caracteristicas": feature_mode,
            "estado": "SIN_DATOS",
        }, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    feature_cols = columnas_de_caracteristicas(subset, feature_mode)
    if not feature_cols:
        return ({
            "ventana_s": window_s,
            "modo_caracteristicas": feature_mode,
            "estado": "SIN_CARACTERISTICAS",
        }, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    subset = subset.dropna(subset=["objetivo", "sujeto"])

    # Los registros válidos se igualan para todos los tamaños de ventana
    # antes de esta función (en main), evitando comparar conjuntos distintos.
    unique_subjects = subset["sujeto"].nunique()
    unique_labels = subset["objetivo"].nunique()
    if unique_subjects < 2:
        return ({
            "ventana_s": window_s,
            "modo_caracteristicas": feature_mode,
            "estado": "LOSO_NO_DISPONIBLE: se necesita al menos 2 sujetos",
            "n_sujetos": unique_subjects,
            "n_muestras": len(subset),
            "n_clases": unique_labels,
        }, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    if unique_labels < 2:
        return ({
            "ventana_s": window_s,
            "modo_caracteristicas": feature_mode,
            "estado": "CLASIFICACION_NO_DISPONIBLE: se necesita al menos 2 clases",
            "n_sujetos": unique_subjects,
            "n_muestras": len(subset),
            "n_clases": unique_labels,
        }, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    X = subset[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = subset["objetivo"].astype(str).to_numpy()
    groups = subset["sujeto"].to_numpy()
    class_labels = np.unique(y)

    model = crear_clasificador()

    logo = LeaveOneGroupOut()
    y_true_all: List[str] = []
    y_pred_all: List[str] = []
    fold_rows: List[dict] = []
    pred_rows: List[dict] = []
    invalid_folds = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        y_train = y[train_idx]
        test_subject = int(groups[test_idx][0])
        if np.unique(y_train).size < 2:
            invalid_folds.append(test_subject)
            continue

        model.fit(X.iloc[train_idx], y_train)
        pred = model.predict(X.iloc[test_idx])
        y_test = y[test_idx]

        fold_acc = accuracy_score(y_test, pred)
        fold_bal_acc = balanced_accuracy_score(y_test, pred)
        # En esta función, las clases globales de esta evaluación LOSO
        # están guardadas en `class_labels`; la predicción del fold se llama `pred`.
        fold_f1 = f1_score(
            y_test,
            pred,
            labels=class_labels,
            average="macro",
            zero_division=0,
        )

        fold_rows.append({
            "ventana_s": window_s,
            "modo_caracteristicas": feature_mode,
            "fold": fold,
            "sujeto_prueba": test_subject,
            "n_muestras_prueba": len(test_idx),
            "accuracy": fold_acc,
            "balanced_accuracy": fold_bal_acc,
            "macro_f1": fold_f1,
            "n_clases_entrenamiento": np.unique(y_train).size,
        })

        for loc, idx in enumerate(test_idx):
            pred_rows.append({
                "ventana_s": window_s,
                "modo_caracteristicas": feature_mode,
                "sujeto": int(groups[idx]),
                "codigo": subset.iloc[idx]["codigo"],
                "objetivo_real": str(y[idx]),
                "objetivo_predicho": str(pred[loc]),
            })

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(pred.tolist())

    if invalid_folds:
        estado = (
            "PARCIAL: algunos folds no tienen al menos dos clases de entrenamiento"
            if fold_rows else
            "NO_EVALUABLE: los folds no contienen dos clases en entrenamiento"
        )
    else:
        estado = "OK"

    if not fold_rows:
        summary = {
            "ventana_s": window_s,
            "modo_caracteristicas": feature_mode,
            "estado": estado,
            "n_sujetos": unique_subjects,
            "n_muestras": len(subset),
            "n_clases": unique_labels,
        }
    else:
        folds = pd.DataFrame(fold_rows)
        summary = {
            "ventana_s": window_s,
            "modo_caracteristicas": feature_mode,
            "estado": estado,
            "n_sujetos": unique_subjects,
            "n_muestras": len(subset),
            "n_clases": unique_labels,
            "n_features": len(feature_cols),
            "accuracy_pooled": accuracy_score(y_true_all, y_pred_all),
            "balanced_accuracy_pooled": balanced_accuracy_score(y_true_all, y_pred_all),
            "macro_f1_pooled": f1_score(
                y_true_all, y_pred_all, labels=class_labels,
                average="macro", zero_division=0
            ),
            "balanced_accuracy_loso_media": folds["balanced_accuracy"].mean(),
            "balanced_accuracy_loso_sd": folds["balanced_accuracy"].std(ddof=1),
            "macro_f1_loso_media": folds["macro_f1"].mean(),
            "macro_f1_loso_sd": folds["macro_f1"].std(ddof=1),
            "folds_no_validos": len(invalid_folds),
        }

    confusion = pd.DataFrame()
    if y_true_all:
        cm = confusion_matrix(y_true_all, y_pred_all, labels=class_labels)
        confusion = pd.DataFrame(
            cm,
            index=[f"real_{c}" for c in class_labels],
            columns=[f"pred_{c}" for c in class_labels],
        )
        confusion.index.name = "clase"

    return summary, pd.DataFrame(fold_rows), pd.DataFrame(pred_rows), confusion


def evaluar_loso_anidado(
    df_features: pd.DataFrame,
    feature_mode: str,
    windows: List[float],
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """LOSO anidado: elige T en sujetos de entrenamiento y lo evalúa en otro sujeto.

    En cada fold externo se elige la duración con mejor macro-F1 de validación
    interna Leave-One-Subject-Out; luego se ajusta el modelo usando todos los
    sujetos de entrenamiento para esa duración y se evalúa en el sujeto externo.
    """
    d = df_features[df_features["modo_caracteristicas"] == feature_mode].copy()
    subjects = np.sort(d["sujeto"].dropna().astype(int).unique())
    if len(subjects) < 3:
        summary = {
            "modo_caracteristicas": feature_mode,
            "estado": "LOSO_ANIDADO_NO_DISPONIBLE: se necesitan al menos 3 sujetos",
            "n_sujetos": len(subjects),
        }
        return summary, pd.DataFrame(), pd.DataFrame()

    d = d[d["ventana_s"].isin([float(w) for w in windows])].copy()
    d = d.dropna(subset=["objetivo", "sujeto"])
    feature_cols = columnas_de_caracteristicas(d, feature_mode)
    if not feature_cols:
        return ({
            "modo_caracteristicas": feature_mode,
            "estado": "SIN_CARACTERISTICAS",
        }, pd.DataFrame(), pd.DataFrame())

    y_global = d["objetivo"].astype(str).to_numpy()
    global_classes = np.unique(y_global)
    folds_nested: List[dict] = []
    preds_nested: List[dict] = []
    y_true_outer: List[str] = []
    y_pred_outer: List[str] = []

    for outer_i, test_subject in enumerate(subjects, start=1):
        d_train_all = d[d["sujeto"].astype(int) != int(test_subject)].copy()
        d_test_subject = d[d["sujeto"].astype(int) == int(test_subject)].copy()
        inner_scores = []

        for window_s in windows:
            d_train_w = d_train_all[d_train_all["ventana_s"] == float(window_s)].copy()
            if d_train_w["sujeto"].nunique() < 2 or d_train_w["objetivo"].nunique() < 2:
                inner_scores.append((float(window_s), np.nan))
                continue

            X_inner = d_train_w[feature_cols].replace([np.inf, -np.inf], np.nan)
            y_inner = d_train_w["objetivo"].astype(str).to_numpy()
            groups_inner = d_train_w["sujeto"].astype(int).to_numpy()
            inner_logo = LeaveOneGroupOut()
            true_inner: List[str] = []
            pred_inner: List[str] = []

            for inner_train_idx, inner_val_idx in inner_logo.split(
                X_inner, y_inner, groups_inner
            ):
                if np.unique(y_inner[inner_train_idx]).size < 2:
                    continue
                inner_model = crear_clasificador()
                inner_model.fit(X_inner.iloc[inner_train_idx], y_inner[inner_train_idx])
                p = inner_model.predict(X_inner.iloc[inner_val_idx])
                true_inner.extend(y_inner[inner_val_idx].tolist())
                pred_inner.extend(p.tolist())

            if true_inner:
                score = f1_score(
                    true_inner, pred_inner, labels=np.unique(y_inner),
                    average="macro", zero_division=0
                )
            else:
                score = np.nan
            inner_scores.append((float(window_s), float(score)))

        valid_scores = [(w, score) for w, score in inner_scores if np.isfinite(score)]
        if not valid_scores:
            folds_nested.append({
                "sujeto_prueba": int(test_subject),
                "estado": "SIN_FOLDS_INTERNOS_VALIDOS",
                "ventana_seleccionada_s": np.nan,
            })
            continue

        # Si hay empate exacto, preferir la ventana más corta para conservar
        # mayor resolución temporal, siempre que el rendimiento sea equivalente.
        max_score = max(score for _, score in valid_scores)
        tied = [w for w, score in valid_scores if np.isclose(score, max_score, atol=1e-12)]
        selected_window = min(tied)

        train_w = d_train_all[d_train_all["ventana_s"] == selected_window].copy()
        test_w = d_test_subject[d_test_subject["ventana_s"] == selected_window].copy()
        if train_w.empty or test_w.empty or train_w["objetivo"].nunique() < 2:
            folds_nested.append({
                "sujeto_prueba": int(test_subject),
                "estado": "SIN_DATOS_EXTERNOS_VALIDOS",
                "ventana_seleccionada_s": selected_window,
                "macro_f1_interno": max_score,
            })
            continue

        X_train = train_w[feature_cols].replace([np.inf, -np.inf], np.nan)
        y_train = train_w["objetivo"].astype(str).to_numpy()
        X_test = test_w[feature_cols].replace([np.inf, -np.inf], np.nan)
        y_test = test_w["objetivo"].astype(str).to_numpy()

        outer_model = crear_clasificador()
        outer_model.fit(X_train, y_train)
        y_pred = outer_model.predict(X_test)
        # En LOSO anidado, `global_classes` se define al inicio de esta función.
        # Se fija la misma lista de clases para evaluar todos los sujetos externos.
        fold_f1 = f1_score(
            y_test,
            y_pred,
            labels=global_classes,
            average="macro",
            zero_division=0,
        )
        fold_bacc = balanced_accuracy_score(y_test, y_pred)

        folds_nested.append({
            "sujeto_prueba": int(test_subject),
            "estado": "OK",
            "ventana_seleccionada_s": selected_window,
            "macro_f1_interno": max_score,
            "macro_f1_externo": fold_f1,
            "balanced_accuracy_externa": fold_bacc,
            "n_muestras_prueba": len(test_w),
        })
        for row_idx, pred in zip(test_w.index, y_pred):
            source = test_w.loc[row_idx]
            preds_nested.append({
                "sujeto": int(source["sujeto"]),
                "codigo": source["codigo"],
                "ventana_seleccionada_s": selected_window,
                "objetivo_real": str(source["objetivo"]),
                "objetivo_predicho": str(pred),
            })
        y_true_outer.extend(y_test.tolist())
        y_pred_outer.extend(y_pred.tolist())

    folds_df = pd.DataFrame(folds_nested)
    pred_df = pd.DataFrame(preds_nested)
    successful = folds_df[folds_df.get("estado", pd.Series(dtype=str)) == "OK"]
    summary = {
        "modo_caracteristicas": feature_mode,
        "estado": "OK" if len(successful) == len(subjects) else "PARCIAL",
        "n_sujetos": len(subjects),
        "n_folds_externos_validos": len(successful),
        "macro_f1_pooled_externo": (
            f1_score(y_true_outer, y_pred_outer, labels=global_classes,
                     average="macro", zero_division=0) if y_true_outer else np.nan
        ),
        "balanced_accuracy_pooled_externa": (
            balanced_accuracy_score(y_true_outer, y_pred_outer) if y_true_outer else np.nan
        ),
        "ventana_seleccionada_mediana_s": (
            successful["ventana_seleccionada_s"].median() if not successful.empty else np.nan
        ),
    }
    return summary, folds_df, pred_df


# =====================================================================
# 8. PROGRAMA PRINCIPAL
# =====================================================================

def main() -> None:
    start_time = time.perf_counter()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    if not RUTA_BASE.exists():
        raise FileNotFoundError(f"No se encuentra RUTA_BASE: {RUTA_BASE}")

    labels = cargar_labels()
    sujetos_a_procesar = sorted(SUJETOS if SUJETOS is not None else descubrir_sujetos())
    if not sujetos_a_procesar:
        raise RuntimeError("No se encontraron carpetas SUB## en RUTA_BASE.")

    print("=" * 78)
    print("EVALUACIÓN DE DURACIÓN DE VENTANAS PARA wPLI EEG")
    print("=" * 78)
    print(f"Sujetos: {sujetos_a_procesar}")
    print(f"Videos: {VIDEOS if VIDEOS is not None else 'todos los disponibles'}")
    print(f"Ventanas: {VENTANAS_SEGUNDOS}")
    print(f"Objetivo LOSO: {TARGET_COLUMN}")
    print(f"Modo(s) de características: {MODOS_CARACTERISTICAS}")
    print(f"Multitaper bandwidth fijo: {MT_BANDWIDTH_HZ} Hz")
    print("No se recortan extremos del registro.")

    rows_features: List[dict] = []
    rows_adequacy: List[dict] = []
    rows_stability: List[dict] = []
    rows_recording_status: List[dict] = []

    # La lista de códigos se basa en archivos disponibles más las etiquetas.
    jobs = []
    for subject in sujetos_a_procesar:
        videos_a_procesar = VIDEOS if VIDEOS is not None else range(1, 100)
        for video in videos_a_procesar:
            code = f"SUB{subject:02d}VID{video:02d}"
            if code not in labels.index:
                continue
            path = localizar_archivo(subject, video)
            if path is not None:
                jobs.append((subject, video, code, path))

    if not jobs:
        raise RuntimeError(
            "No se encontraron experimentos que coincidan con archivos .set y labels.csv."
        )

    print(f"Experimentos encontrados con etiqueta: {len(jobs)}")

    for subject, video, code, path in jobs:
        print("\n" + "-" * 78)
        print(f"Procesando {code}: {path.name}")
        target = labels.loc[code, TARGET_COLUMN]
        if pd.isna(target):
            print(f"  Etiqueta {TARGET_COLUMN} vacía; se omite.")
            continue

        try:
            raw_clean, ica_excluded = preparar_eeg(path)
        except Exception as exc:
            print(f"  ERROR en preprocesamiento {code}: {exc}")
            rows_recording_status.append({
                "codigo": code, "sujeto": subject, "video": video,
                "estado": f"ERROR_PREPROCESAMIENTO: {exc}",
            })
            continue

        duration = float(raw_clean.times[-1])
        sfreq = float(raw_clean.info["sfreq"])
        channels = list(raw_clean.ch_names)
        print(
            f"  Canales={len(channels)}, sfreq={sfreq:.2f} Hz, "
            f"duración={duration:.2f} s, ICA excluidos={ica_excluded}"
        )

        for window_s in VENTANAS_SEGUNDOS:
            # Cada duración produce una partición diferente del mismo registro.
            try:
                epochs = mne.make_fixed_length_epochs(
                    raw_clean,
                    duration=float(window_s),
                    preload=True,
                    reject_by_annotation=True,
                    verbose=False,
                )
            except Exception as exc:
                print(f"  ERROR epoching {code}, T={window_s}: {exc}")
                rows_recording_status.append({
                    "codigo": code, "sujeto": subject, "video": video,
                    "ventana_s": window_s,
                    "estado": f"ERROR_EPOCHING: {exc}",
                })
                continue

            n_epochs = len(epochs)
            rows_adequacy.append(
                registro_adecuacion(
                    code, subject, video, duration, float(window_s), sfreq, n_epochs
                )
            )
            print(f"  T={window_s:g} s -> {n_epochs} épocas")

            if n_epochs < MIN_EPOCAS_WPLI:
                print(
                    f"    Se omite wPLI: {n_epochs} épocas < "
                    f"MIN_EPOCAS_WPLI={MIN_EPOCAS_WPLI}."
                )
                rows_recording_status.append({
                    "codigo": code, "sujeto": subject, "video": video,
                    "ventana_s": window_s, "n_epocas": n_epochs,
                    "estado": "INSUFICIENTES_EPOCAS_WPLI",
                })
                continue

            # Matriz completa del registro: base de características.
            try:
                matrices = calcular_matrices_wpli(epochs)
            except Exception as exc:
                print(f"    ERROR wPLI {code}, T={window_s}: {exc}")
                rows_recording_status.append({
                    "codigo": code, "sujeto": subject, "video": video,
                    "ventana_s": window_s, "n_epocas": n_epochs,
                    "estado": f"ERROR_WPLI: {exc}",
                })
                continue

            common_meta = {
                "codigo": code,
                "sujeto": subject,
                "video": video,
                "ventana_s": float(window_s),
                "n_epocas": n_epochs,
                "objetivo": str(target),
            }

            for feature_mode in MODOS_CARACTERISTICAS:
                if feature_mode == "edges":
                    feats = feature_edges(matrices, channels)
                elif feature_mode == "graph":
                    feats = graph_summary_features(matrices, channels)
                else:
                    raise ValueError(
                        f"Modo desconocido: {feature_mode}. Usa 'edges' o 'graph'."
                    )
                rows_features.append({
                    **common_meta,
                    "modo_caracteristicas": feature_mode,
                    **feats,
                })

            # Estabilidad interna estimada con split-half repetido.
            if CALCULAR_ESTABILIDAD and N_SPLIT_REPEATS > 0:
                estabilidad = calcular_estabilidad_split_half(
                    epochs, code, float(window_s), subject, video
                )
                rows_stability.extend(estabilidad)

            del epochs, matrices

        del raw_clean

    # -----------------------------------------------------------------
    # Guardar datos base
    # -----------------------------------------------------------------
    if not rows_features:
        raise RuntimeError(
            "No se generaron características. Revisar rutas, etiquetas y errores del log."
        )

    df_features = pd.DataFrame(rows_features)
    df_adequacy = pd.DataFrame(rows_adequacy)
    df_stability = pd.DataFrame(rows_stability)
    df_status = pd.DataFrame(rows_recording_status)

    df_adequacy.to_csv(CARPETA_SALIDA / "adecuacion_espectral_por_registro.csv", index=False)
    df_features.to_csv(CARPETA_SALIDA / "caracteristicas_por_registro.csv", index=False)
    if not df_stability.empty:
        df_stability.to_csv(CARPETA_SALIDA / "estabilidad_split_half_detalle.csv", index=False)
    if not df_status.empty:
        df_status.to_csv(CARPETA_SALIDA / "incidencias_procesamiento.csv", index=False)

    # -----------------------------------------------------------------
    # Para comparar la clasificación, utilizar la INTERSECCIÓN de códigos
    # con features válidas en TODAS las duraciones. Esto evita que una T
    # resulte aparentemente mejor sólo porque usa ejemplos distintos.
    # -----------------------------------------------------------------
    expected_windows = set(float(w) for w in VENTANAS_SEGUNDOS)
    mode_sample_sets = {}
    for mode in MODOS_CARACTERISTICAS:
        d = df_features[df_features["modo_caracteristicas"] == mode]
        sample_sets = []
        for w in VENTANAS_SEGUNDOS:
            codes = set(
                d.loc[
                    (d["ventana_s"] == float(w))
                    & (d["n_epocas"] >= MIN_EPOCAS_CLASIFICACION),
                    "codigo",
                ].astype(str)
            )
            sample_sets.append(codes)
        common_codes = set.intersection(*sample_sets) if sample_sets else set()
        mode_sample_sets[mode] = common_codes
        print(
            f"\nCódigos comunes para clasificación ({mode}): "
            f"{len(common_codes)}"
        )

    # -----------------------------------------------------------------
    # LOSO por duración y tipo de características
    # -----------------------------------------------------------------
    cls_summaries = []
    all_fold_rows = []
    all_prediction_rows = []
    confusion_tables = []

    for mode in MODOS_CARACTERISTICAS:
        common_codes = mode_sample_sets[mode]
        d_mode = df_features[df_features["modo_caracteristicas"] == mode].copy()
        d_mode = d_mode[
            d_mode["codigo"].astype(str).isin(common_codes)
            & (d_mode["n_epocas"] >= MIN_EPOCAS_CLASIFICACION)
        ].copy()

        for window_s in VENTANAS_SEGUNDOS:
            result = evaluar_loso(d_mode, float(window_s), mode)
            # La función devuelve 4 objetos cuando finaliza normalmente.
            summary, fold_df, pred_df, cm_df = result
            cls_summaries.append(summary)
            if not fold_df.empty:
                all_fold_rows.append(fold_df)
            if not pred_df.empty:
                all_prediction_rows.append(pred_df)
            if not cm_df.empty:
                cm_df.to_csv(
                    CARPETA_SALIDA
                    / f"confusion_{mode}_{str(window_s).replace('.', 'p')}s.csv"
                )

    df_cls = pd.DataFrame(cls_summaries)
    df_cls.to_csv(CARPETA_SALIDA / "resultados_clasificacion_LOSO.csv", index=False)
    if all_fold_rows:
        pd.concat(all_fold_rows, ignore_index=True).to_csv(
            CARPETA_SALIDA / "resultados_LOSO_por_sujeto.csv", index=False
        )
    if all_prediction_rows:
        pd.concat(all_prediction_rows, ignore_index=True).to_csv(
            CARPETA_SALIDA / "predicciones_LOSO.csv", index=False
        )

    # -----------------------------------------------------------------
    # LOSO ANIDADO opcional: evaluación más adecuada si se selecciona T
    # automáticamente a partir del desempeño de validación.
    # -----------------------------------------------------------------
    if CALCULAR_LOSO_ANIDADO:
        nested_summaries = []
        nested_fold_tables = []
        nested_prediction_tables = []
        for mode in MODOS_CARACTERISTICAS:
            common_codes = mode_sample_sets[mode]
            d_mode = df_features[
                (df_features["modo_caracteristicas"] == mode)
                & df_features["codigo"].astype(str).isin(common_codes)
                & (df_features["n_epocas"] >= MIN_EPOCAS_CLASIFICACION)
            ].copy()
            summary, folds_df, preds_df = evaluar_loso_anidado(
                d_mode, mode, [float(w) for w in VENTANAS_SEGUNDOS]
            )
            nested_summaries.append(summary)
            if not folds_df.empty:
                nested_fold_tables.append(folds_df.assign(modo_caracteristicas=mode))
            if not preds_df.empty:
                nested_prediction_tables.append(preds_df.assign(modo_caracteristicas=mode))
        pd.DataFrame(nested_summaries).to_csv(
            CARPETA_SALIDA / "resultados_LOSO_anidado_seleccion_ventana.csv", index=False
        )
        if nested_fold_tables:
            pd.concat(nested_fold_tables, ignore_index=True).to_csv(
                CARPETA_SALIDA / "folds_LOSO_anidado.csv", index=False
            )
        if nested_prediction_tables:
            pd.concat(nested_prediction_tables, ignore_index=True).to_csv(
                CARPETA_SALIDA / "predicciones_LOSO_anidado.csv", index=False
            )

    # -----------------------------------------------------------------
    # Resumen de adecuación espectral por duración
    # -----------------------------------------------------------------
    adequate_summary = (
        df_adequacy.groupby("ventana_s", as_index=False)
        .agg(
            n_registros=("codigo", "nunique"),
            epocas_mediana=("n_epocas", "median"),
            epocas_min=("n_epocas", "min"),
            epocas_max=("n_epocas", "max"),
            ciclos_4hz=("ciclos_a_fmin_minimo", "first"),
            resolucion_nominal_hz=("resolucion_nominal_1_sobre_T_hz", "first"),
            fraccion_registros_ge_5_ciclos=("al_menos_5_ciclos_a_4Hz", "mean"),
            fraccion_registros_con_epocas_suficientes=("n_epocas_suficientes_clasificacion", "mean"),
        )
        .sort_values("ventana_s")
    )
    adequate_summary.to_csv(
        CARPETA_SALIDA / "resumen_adecuacion_espectral_por_ventana.csv", index=False
    )

    # -----------------------------------------------------------------
    # Resumen de estabilidad
    # -----------------------------------------------------------------
    if not df_stability.empty:
        successful = df_stability[df_stability["estado"] == "OK"].copy()
        if not successful.empty:
            stability_summary = (
                successful.groupby(["ventana_s", "banda"], as_index=False)
                .agg(
                    n_comparaciones=("correlacion_aristas", "count"),
                    correlacion_mediana=("correlacion_aristas", "median"),
                    correlacion_media=("correlacion_aristas", "mean"),
                    correlacion_sd=("correlacion_aristas", "std"),
                    rmse_mediano=("rmse_aristas", "median"),
                    rmse_medio=("rmse_aristas", "mean"),
                    mae_mediano=("mae_aristas", "median"),
                    nrmse_mediano=("nrmse_aristas", "median"),
                )
                .sort_values(["ventana_s", "banda"])
            )
            stability_summary.to_csv(
                CARPETA_SALIDA / "resumen_estabilidad_por_ventana_banda.csv", index=False
            )
        else:
            print("\nNo hubo suficientes particiones para resumir estabilidad.")

    # -----------------------------------------------------------------
    # Gráficas comparativas
    # -----------------------------------------------------------------
    if not df_stability.empty:
        successful = df_stability[df_stability["estado"] == "OK"].copy()
        if not successful.empty:
            import matplotlib.pyplot as plt

            grouped = successful.groupby("ventana_s")["correlacion_aristas"].agg(["median", "mean", "std"])
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(grouped.index, grouped["median"], marker="o", label="Mediana")
            ax.plot(grouped.index, grouped["mean"], marker="s", label="Media")
            ax.set_xlabel("Duración de ventana (s)")
            ax.set_ylabel("Correlación entre matrices split-half")
            ax.set_title("Estabilidad interna de wPLI según duración de ventana")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(CARPETA_SALIDA / "estabilidad_wpli_vs_ventana.png", dpi=180)
            plt.close(fig)

    # Clasificación por ventana
    if not df_cls.empty and "macro_f1_pooled" in df_cls.columns:
        rows_to_plot = df_cls[
            df_cls["estado"].astype(str).str.startswith(("OK", "PARCIAL"))
        ].copy()
        if not rows_to_plot.empty:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            for mode in MODOS_CARACTERISTICAS:
                d = rows_to_plot[rows_to_plot["modo_caracteristicas"] == mode].sort_values("ventana_s")
                if not d.empty and "macro_f1_pooled" in d:
                    ax.plot(d["ventana_s"], d["macro_f1_pooled"], marker="o", label=mode)
            ax.set_xlabel("Duración de ventana (s)")
            ax.set_ylabel("Macro-F1 pooled LOSO")
            ax.set_title("Utilidad de clasificación LOSO según ventana")
            ax.grid(alpha=0.3)
            ax.legend(title="Características")
            fig.tight_layout()
            fig.savefig(CARPETA_SALIDA / "macro_f1_LOSO_vs_ventana.png", dpi=180)
            plt.close(fig)

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 78)
    print("EVALUACIÓN TERMINADA")
    print("=" * 78)
    print(f"Resultados: {CARPETA_SALIDA}")
    print(f"Tiempo total: {elapsed:.1f} s")
    print("\nResumen de adecuación espectral:")
    print(adequate_summary.to_string(index=False))
    print("\nResumen de clasificación LOSO:")
    print(df_cls.to_string(index=False))
    print(
        "\nNota: si se configuró un único sujeto, la estabilidad y adecuación "
        "se calculan, pero LOSO queda correctamente marcado como no disponible."
    )


if __name__ == "__main__":
    main()
