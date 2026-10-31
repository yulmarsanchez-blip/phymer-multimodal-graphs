# -*- coding: utf-8 -*-
"""
Fiabilidad de la entropía espectral de grafos EEG construidos con wPLI.

Basado en el preprocesamiento del script evaluacion_ventanas_wpli_corregido.py:
  1. Leer EEGLAB .set y seleccionar EEG.
  2. Ajustar ICA en una copia filtrada 1-45 Hz.
  3. Detectar componentes oculares usando AF3/AF4 como proxies EOG.
  4. Aplicar ICA a la señal original sin el filtro FIR 1-45 Hz.
  5. Referencia promedio.
  6. Crear épocas contiguas de duración fija, sin recortar extremos.
  7. Calcular wPLI multitaper por bandas (Theta, Alpha, Beta, Gamma).

Para cada registro, duración, banda y partición aleatoria repetida:
  - divide las épocas en dos mitades disjuntas;
  - calcula una matriz wPLI por mitad;
  - calcula entropía espectral laplaciana por mitad;
  - compara las dos estimaciones.

Entropía definida aquí:
  L = D - W, p_i = lambda_i / sum(lambda_j)
  H = -sum_i p_i log(p_i) [nats]
  H_norm = H / log(N)

Esta es una entropía espectral de la laplaciana normalizada por su traza.
No debe interpretarse automáticamente como entropía de la señal EEG ni como
una medida universal de complejidad cerebral.

Se usa una cohorte común: un registro entra al análisis principal solo si
contiene suficientes épocas para dividirse en dos mitades en TODAS las
ventanas configuradas. Esto evita comparar duraciones con conjuntos distintos
de registros por causa de la disponibilidad de épocas.

Archivos de salida:
  - entropia_espectral_split_half_detalle.csv
  - fiabilidad_entropia_por_ventana_banda_repeticion.csv
  - fiabilidad_entropia_resumen_por_ventana_banda.csv
  - incidencias_entropia_split_half.csv
  - fiabilidad_entropia_correlacion.png
  - fiabilidad_entropia_error.png
  - configuracion_entropia.txt

Nota estadística: las métricas por repetición se calculan entre registros
(sujeto-video). Los vídeos de un mismo sujeto no son independientes; por eso
no se reportan p-valores ingenuos para las correlaciones. Pearson/Spearman,
MAE/RMSE e ICC son descriptores de reproducibilidad y deben interpretarse en
conjunto. Las repeticiones split-half tampoco son observaciones independientes.
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
from mne_connectivity import spectral_connectivity_epochs
from scipy.stats import pearsonr, spearmanr


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
) / "fiabilidad_entropia_espectral"

SUJETOS = list(range(1, 31))
VIDEOS = list(range(1, 26))
VENTANAS_SEGUNDOS = [1.0, 2.0, 3.0, 4.0, 5.0]

# Preprocesamiento ICA, exactamente el criterio del script de evaluación.
ICA_L_FREQ = 1.0
ICA_H_FREQ = 45.0
ICA_RANDOM_STATE = 42
ICA_EOG_THRESHOLD = 3.0

BANDAS = {
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 40.0),
}
BAND_NAMES = list(BANDAS.keys())
FMIN = [BANDAS[b][0] for b in BAND_NAMES]
FMAX = [BANDAS[b][1] for b in BAND_NAMES]
MT_BANDWIDTH_HZ = 2.0
N_JOBS_WPLI = 1

# Mínimo de épocas EN CADA mitad. Total mínimo por registro = 2 * este valor.
# 5 mantiene la comparabilidad con el análisis previo, pero es un umbral
# permisivo. Para un análisis de sensibilidad más estricto, probar 8.
MIN_EPOCAS_POR_MITAD = 5

# Número de particiones aleatorias por registro y duración.
# 5 es un compromiso entre tiempo de ejecución y variabilidad del split.
# Para una ejecución final de sensibilidad Monte Carlo, se puede probar 10.
N_SPLIT_REPEATS = 5
RANDOM_STATE = 20261013

N_NODOS_ESPERADOS = 14
CANALES_ESPERADOS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]
EPS = 1e-12

COLUMNAS_ETIQUETAS_OPCIONALES = ["emotion", "arousal", "valence"]


# =====================================================================
# 2. ARCHIVOS, ETIQUETAS Y PREPROCESAMIENTO
# =====================================================================

def extraer_codigo(path: Path) -> Optional[str]:
    """Extrae un identificador como SUB01VID01 del nombre del archivo."""
    match = re.search(r"(SUB\d+VID\d+)", path.stem, re.IGNORECASE)
    return match.group(1).upper() if match else None


def localizar_archivo(sujeto: int, video: int) -> Optional[Path]:
    """Busca el .set y prefiere el nombre convencional SUB##VID##_eeg.set."""
    codigo = f"SUB{sujeto:02d}VID{video:02d}"
    candidatos = list(RUTA_BASE.rglob(f"{codigo}*_eeg*.set"))
    if not candidatos:
        candidatos = list(RUTA_BASE.rglob(f"{codigo}*.set"))
        candidatos = [p for p in candidatos if extraer_codigo(p) == codigo]
    if not candidatos:
        return None

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
            f"  AVISO: {codigo} tiene {len(candidatos)} candidatos; "
            f"se usará {candidatos[0].name}"
        )
    return candidatos[0]


def cargar_labels() -> pd.DataFrame:
    """Carga etiquetas; no exige que emotion/arousal/valence estén completos."""
    if not ARCHIVO_LABELS.exists():
        raise FileNotFoundError(f"No se encuentra labels.csv: {ARCHIVO_LABELS}")
    labels = pd.read_csv(ARCHIVO_LABELS)
    if "experiment_code" not in labels.columns:
        raise ValueError(
            "labels.csv debe contener la columna 'experiment_code'. "
            f"Columnas encontradas: {list(labels.columns)}"
        )
    labels = labels.copy()
    labels["experiment_code"] = (
        labels["experiment_code"].astype(str).str.strip().str.upper()
    )
    duplicados = labels.loc[
        labels["experiment_code"].duplicated(keep=False), "experiment_code"
    ].unique()
    if len(duplicados):
        raise ValueError(
            "Hay experiment_code duplicados en labels.csv. Ejemplos: "
            f"{duplicados[:5].tolist()}"
        )
    return labels.set_index("experiment_code", drop=False)


def obtener_metadatos_etiqueta(labels: pd.DataFrame, codigo: str) -> dict:
    """Devuelve emotion/arousal/valence si existen en la tabla de etiquetas."""
    if codigo not in labels.index:
        return {}
    row = labels.loc[codigo]
    result = {}
    for col in COLUMNAS_ETIQUETAS_OPCIONALES:
        if col in labels.columns and pd.notna(row[col]):
            result[col] = row[col]
    return result


def preparar_eeg(ruta: Path) -> Tuple[mne.io.BaseRaw, List[int]]:
    """Preprocesamiento replicado del script anterior de evaluación de ventanas."""
    raw = mne.io.read_raw_eeglab(str(ruta), preload=True, verbose=False)
    raw.pick("eeg")

    if len(raw.ch_names) < 4:
        raise ValueError(
            f"Muy pocos canales EEG tras raw.pick('eeg'): {raw.ch_names}"
        )

    # ICA se ajusta sobre una copia filtrada 1-45 Hz.
    # El filtro no se aplica al Raw limpio sobre el que se estima wPLI.
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
            print(f"  AVISO: fallo detección proxy EOG en AF3: {exc}")
    if "AF4" in raw_ica.ch_names:
        try:
            componentes_af4, _ = ica.find_bads_eog(
                raw_ica,
                ch_name="AF4",
                threshold=ICA_EOG_THRESHOLD,
                verbose=False,
            )
        except Exception as exc:
            print(f"  AVISO: fallo detección proxy EOG en AF4: {exc}")

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
# 3. wPLI: MISMO ESTIMADOR QUE EL SCRIPT ANTERIOR
# =====================================================================

def simetrizar_wpli(matrix: np.ndarray) -> np.ndarray:
    """Devuelve una matriz wPLI simétrica, sin diagonal y sin pesos negativos."""
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
    return np.clip(out, 0.0, None)


def calcular_matrices_wpli(epochs: mne.Epochs) -> Dict[str, np.ndarray]:
    """Calcula una matriz wPLI por banda usando multitaper y faverage=True."""
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
        band: simetrizar_wpli(densa[:, :, idx])
        for idx, band in enumerate(BAND_NAMES)
    }


# =====================================================================
# 4. ENTROPÍA ESPECTRAL DE LA LAPLACIANA
# =====================================================================

def entropia_espectral_laplaciana(matrix: np.ndarray) -> Tuple[float, float, float]:
    """Devuelve (H_nats, H_normalizada, wPLI_media_de_aristas).

    Definición: L = D-W; p_i = lambda_i / sum(lambda); H=-sum(p_i ln p_i).
    H_normalizada = H / ln(N), donde N es el número de nodos.

    La entropía es invariante ante multiplicar todos los pesos por una
    constante positiva; por eso también se guarda la wPLI media de aristas,
    que resume la escala de los pesos y no debe confundirse con la entropía.
    """
    w = np.asarray(matrix, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] < 2:
        return np.nan, np.nan, np.nan

    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = 0.5 * (w + w.T)
    np.fill_diagonal(w, 0.0)
    w = np.clip(w, 0.0, None)

    n_nodes = w.shape[0]
    edge_values = w[np.triu_indices(n_nodes, k=1)]
    mean_edge_weight = float(np.mean(edge_values)) if edge_values.size else np.nan

    degree = np.sum(w, axis=1)
    laplacian = np.diag(degree) - w
    eigenvalues = np.linalg.eigvalsh(laplacian)

    # Los autovalores de una laplaciana no negativa deberían ser >= 0;
    # se eliminan pequeños negativos exclusivamente por redondeo numérico.
    eigenvalues = np.where(eigenvalues < 0.0, np.maximum(eigenvalues, -1e-10), eigenvalues)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(np.sum(eigenvalues))
    if not np.isfinite(total) or total <= EPS:
        return np.nan, np.nan, mean_edge_weight

    probabilities = eigenvalues / total
    positive = probabilities > EPS
    entropy_nats = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    entropy_normalized = float(entropy_nats / np.log(n_nodes))
    return entropy_nats, entropy_normalized, mean_edge_weight


# =====================================================================
# 5. MÉTRICAS DE REPRODUCIBILIDAD
# =====================================================================

def correlacion_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3 or np.std(a) < EPS or np.std(b) < EPS:
        return np.nan
    return float(pearsonr(a, b)[0])


def correlacion_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3 or np.std(a) < EPS or np.std(b) < EPS:
        return np.nan
    return float(spearmanr(a, b)[0])


def icc_acuerdo_absoluto_a1(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(A,1): ICC de dos vías, acuerdo absoluto, medida individual.

    Las filas son registros (unidades) y las columnas son las estimaciones
    de las dos mitades. El cálculo supone una matriz completa de dos medidas
    por unidad. Se usa como descriptor, con cautela por las repeticiones de
    vídeo dentro de cada sujeto.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    y = np.column_stack((a[valid], b[valid]))
    n, k = y.shape if y.ndim == 2 else (0, 0)
    if n < 3 or k != 2:
        return np.nan

    grand_mean = float(np.mean(y))
    mean_by_record = np.mean(y, axis=1)
    mean_by_half = np.mean(y, axis=0)

    ms_record = k * np.sum((mean_by_record - grand_mean) ** 2) / (n - 1)
    ms_half = n * np.sum((mean_by_half - grand_mean) ** 2) / (k - 1)
    residuals = (
        y
        - mean_by_record[:, np.newaxis]
        - mean_by_half[np.newaxis, :]
        + grand_mean
    )
    ms_error = np.sum(residuals ** 2) / ((n - 1) * (k - 1))

    denominator = ms_record + (k - 1) * ms_error + k * (ms_half - ms_error) / n
    if abs(denominator) < EPS:
        return np.nan
    return float((ms_record - ms_error) / denominator)


def resumir_repeticion(detalle: pd.DataFrame) -> pd.DataFrame:
    """Calcula fiabilidad entre registros por ventana, banda y repetición."""
    rows = []
    keys = ["ventana_s", "banda", "repeticion"]
    for (window_s, band, rep), group in detalle.groupby(keys, sort=True):
        good = group.dropna(subset=["H_A_norm", "H_B_norm"]).copy()
        a = good["H_A_norm"].to_numpy(float)
        b = good["H_B_norm"].to_numpy(float)
        delta = a - b
        n = len(good)
        rows.append({
            "ventana_s": float(window_s),
            "banda": band,
            "repeticion": int(rep),
            "n_registros": n,
            "pearson_r": correlacion_pearson(a, b),
            "spearman_rho": correlacion_spearman(a, b),
            "mae_H_norm": float(np.mean(np.abs(delta))) if n else np.nan,
            "rmse_H_norm": float(np.sqrt(np.mean(delta ** 2))) if n else np.nan,
            "sesgo_medio_H_norm_A_menos_B": float(np.mean(delta)) if n else np.nan,
            "mae_H_nats": float(np.mean(np.abs(good["H_A_nats"] - good["H_B_nats"]))) if n else np.nan,
            "rmse_H_nats": float(np.sqrt(np.mean((good["H_A_nats"] - good["H_B_nats"]) ** 2))) if n else np.nan,
            "icc_A1_acuerdo_absoluto": icc_acuerdo_absoluto_a1(a, b),
        })
    return pd.DataFrame(rows)


def resumir_repeticiones(resumen_rep: pd.DataFrame) -> pd.DataFrame:
    """Resume las métricas sobre las particiones, sin tratarlas como independientes."""
    metricas = [
        "pearson_r", "spearman_rho", "mae_H_norm", "rmse_H_norm",
        "sesgo_medio_H_norm_A_menos_B", "mae_H_nats", "rmse_H_nats",
        "icc_A1_acuerdo_absoluto", "n_registros",
    ]
    rows = []
    for (window_s, band), group in resumen_rep.groupby(["ventana_s", "banda"], sort=True):
        row = {
            "ventana_s": float(window_s),
            "banda": band,
            "n_repeticiones": int(group["repeticion"].nunique()),
        }
        for metric in metricas:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_media_entre_splits"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_sd_entre_splits"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            row[f"{metric}_mediana_entre_splits"] = float(vals.median()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["ventana_s", "banda"]).reset_index(drop=True)


# =====================================================================
# 6. PROGRAMA PRINCIPAL
# =====================================================================

def main() -> None:
    start = time.perf_counter()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    if not RUTA_BASE.exists():
        raise FileNotFoundError(
            f"No se encuentra RUTA_BASE: {RUTA_BASE}\n"
            "Edita RUTA_BASE y ARCHIVO_LABELS al inicio del script."
        )

    labels = cargar_labels()
    jobs = []
    status_rows: List[dict] = []
    videos_a_buscar = VIDEOS if VIDEOS is not None else range(1, 100)
    sujetos_a_buscar = SUJETOS if SUJETOS is not None else range(1, 100)

    for subject in sujetos_a_buscar:
        for video in videos_a_buscar:
            code = f"SUB{subject:02d}VID{video:02d}"
            if code not in labels.index:
                continue
            path = localizar_archivo(subject, video)
            if path is None:
                status_rows.append({
                    "codigo": code,
                    "sujeto": subject,
                    "video": video,
                    "estado": "SIN_ARCHIVO_SET",
                })
            else:
                jobs.append((subject, video, code, path))

    if not jobs:
        raise RuntimeError("No se encontraron .set con código presente en labels.csv.")

    print("=" * 80)
    print("FIABILIDAD DE ENTROPÍA ESPECTRAL DE GRAFOS EEG (wPLI)")
    print("=" * 80)
    print(f"Registros con archivo y código etiquetado: {len(jobs)}")
    print(f"Sujetos configurados: {SUJETOS}")
    print(f"Ventanas: {VENTANAS_SEGUNDOS}")
    print(f"Bandas: {BANDAS}")
    print(f"Multitaper bandwidth fijo: {MT_BANDWIDTH_HZ} Hz")
    print(f"Repeticiones split-half: {N_SPLIT_REPEATS}")
    print(f"Mínimo de épocas por mitad: {MIN_EPOCAS_POR_MITAD}")
    print("Cohorte principal: intersección de registros elegibles en todas las ventanas.")
    print("No se recortan extremos y no se aplica el FIR de ICA al Raw usado para wPLI.")

    detalle_rows: List[dict] = []
    cohort_codes: List[str] = []

    for subject, video, code, path in jobs:
        print("\n" + "-" * 80)
        print(f"Procesando {code}: {path.name}")
        try:
            raw_clean, ica_excluded = preparar_eeg(path)
        except Exception as exc:
            print(f"  ERROR preprocesamiento: {exc}")
            status_rows.append({
                "codigo": code, "sujeto": subject, "video": video,
                "estado": f"ERROR_PREPROCESAMIENTO: {exc}",
            })
            continue

        sfreq = float(raw_clean.info["sfreq"])
        channels = list(raw_clean.ch_names)
        if len(channels) != N_NODOS_ESPERADOS or set(channels) != set(CANALES_ESPERADOS):
            status_rows.append({
                "codigo": code, "sujeto": subject, "video": video,
                "n_canales": len(channels),
                "canales": ";".join(channels),
                "estado": "EXCLUIDO_CANALERA_DISTINTA_DE_LA_ESPERADA",
            })
            print(
                "  Se excluye: los canales no coinciden con el conjunto EEG esperado. "
                f"Detectados: {channels}"
            )
            del raw_clean
            continue

        # Creamos primero todas las particiones de ventana para poder aplicar
        # el criterio de cohorte común sin dar preferencia a una duración.
        epochs_by_window: Dict[float, mne.Epochs] = {}
        counts: Dict[float, int] = {}
        epoch_error = None
        for window_s in VENTANAS_SEGUNDOS:
            try:
                ep = mne.make_fixed_length_epochs(
                    raw_clean,
                    duration=float(window_s),
                    preload=True,
                    reject_by_annotation=True,
                    verbose=False,
                )
                epochs_by_window[float(window_s)] = ep
                counts[float(window_s)] = len(ep)
            except Exception as exc:
                epoch_error = f"ERROR_EPOCHING_T{window_s:g}: {exc}"
                break

        if epoch_error is not None:
            print(f"  {epoch_error}")
            status_rows.append({
                "codigo": code, "sujeto": subject, "video": video,
                "estado": epoch_error,
            })
            del raw_clean
            continue

        min_total = 2 * MIN_EPOCAS_POR_MITAD
        deficient = [w for w in VENTANAS_SEGUNDOS if counts[float(w)] < min_total]
        counts_text = ";".join(f"T{w:g}s={counts[float(w)]}" for w in VENTANAS_SEGUNDOS)
        print(f"  Canales={len(channels)}, fs={sfreq:.2f} Hz, ICA excluidos={ica_excluded}")
        print(f"  Épocas: {counts_text}")

        if deficient:
            deficient_text = ";".join(
                f"T{w:g}s={counts[float(w)]}" for w in deficient
            )
            print(
                f"  Se excluye de TODAS las ventanas para mantener cohorte común; "
                f"no alcanza {min_total} épocas en: {deficient_text}"
            )
            status_rows.append({
                "codigo": code,
                "sujeto": subject,
                "video": video,
                "n_epocas_por_ventana": counts_text,
                "ventanas_deficientes": deficient_text,
                "estado": "EXCLUIDO_COHORTE_COMUN_POR_EPOCAS_INSUFICIENTES",
            })
            del epochs_by_window, raw_clean
            continue

        cohort_codes.append(code)
        label_meta = obtener_metadatos_etiqueta(labels, code)

        for window_s in VENTANAS_SEGUNDOS:
            window_s = float(window_s)
            epochs = epochs_by_window[window_s]
            n_epochs = len(epochs)
            n_a_expected = n_epochs // 2
            n_b_expected = n_epochs - n_a_expected
            print(
                f"  T={window_s:g}s: {n_epochs} épocas; "
                f"mitades {n_a_expected}/{n_b_expected}"
            )
            seed = (
                RANDOM_STATE
                + subject * 100_000
                + video * 1_000
                + int(round(window_s * 10))
            )
            rng = np.random.default_rng(seed)
            all_indices = np.arange(n_epochs)

            for repetition in range(1, N_SPLIT_REPEATS + 1):
                permutation = rng.permutation(all_indices)
                idx_a = permutation[:n_a_expected]
                idx_b = permutation[n_a_expected:]
                epochs_a = epochs[idx_a]
                epochs_b = epochs[idx_b]

                try:
                    matrices_a = calcular_matrices_wpli(epochs_a)
                    matrices_b = calcular_matrices_wpli(epochs_b)
                except Exception as exc:
                    print(
                        f"  AVISO: wPLI falló en {code}, T={window_s:g}s, "
                        f"repetición={repetition}: {exc}"
                    )
                    status_rows.append({
                        "codigo": code, "sujeto": subject, "video": video,
                        "ventana_s": window_s, "repeticion": repetition,
                        "estado": f"ERROR_WPLI_SPLIT_HALF: {exc}",
                    })
                    continue

                for band in BAND_NAMES:
                    h_a_nats, h_a_norm, wpli_a_mean = entropia_espectral_laplaciana(
                        matrices_a[band]
                    )
                    h_b_nats, h_b_norm, wpli_b_mean = entropia_espectral_laplaciana(
                        matrices_b[band]
                    )
                    valid_entropy = np.isfinite(h_a_norm) and np.isfinite(h_b_norm)
                    row = {
                        "codigo": code,
                        "sujeto": subject,
                        "video": video,
                        "ventana_s": window_s,
                        "banda": band,
                        "repeticion": repetition,
                        "n_epocas_total": n_epochs,
                        "n_epocas_A": len(idx_a),
                        "n_epocas_B": len(idx_b),
                        "H_A_nats": h_a_nats,
                        "H_B_nats": h_b_nats,
                        "H_A_norm": h_a_norm,
                        "H_B_norm": h_b_norm,
                        "diferencia_H_norm_A_menos_B": h_a_norm - h_b_norm if valid_entropy else np.nan,
                        "diferencia_absoluta_H_norm": abs(h_a_norm - h_b_norm) if valid_entropy else np.nan,
                        "diferencia_H_nats_A_menos_B": h_a_nats - h_b_nats if np.isfinite(h_a_nats) and np.isfinite(h_b_nats) else np.nan,
                        "diferencia_absoluta_H_nats": abs(h_a_nats - h_b_nats) if np.isfinite(h_a_nats) and np.isfinite(h_b_nats) else np.nan,
                        "wpli_media_A": wpli_a_mean,
                        "wpli_media_B": wpli_b_mean,
                        "estado": "OK" if valid_entropy else "ENTROPIA_NO_DEFINIDA_MATRIZ_NULA",
                        **label_meta,
                    }
                    detalle_rows.append(row)

            del epochs_a, epochs_b

        del epochs_by_window, raw_clean

    if not detalle_rows:
        raise RuntimeError(
            "No se generaron entropías. Revisa rutas, dependencias y los errores de procesamiento."
        )

    df_detail = pd.DataFrame(detalle_rows)
    df_detail.to_csv(
        CARPETA_SALIDA / "entropia_espectral_split_half_detalle.csv", index=False
    )

    # Una métrica por ventana-banda-repetición, calculada sobre los registros
    # para los que se obtuvieron ambas entropías.
    df_repeat = resumir_repeticion(df_detail)
    df_repeat.to_csv(
        CARPETA_SALIDA / "fiabilidad_entropia_por_ventana_banda_repeticion.csv",
        index=False,
    )

    # Media/SD/mediana a través de particiones split-half repetidas. Estas
    # repeticiones son una medida de sensibilidad a la partición, no N nuevos.
    df_summary = resumir_repeticiones(df_repeat)
    df_summary.to_csv(
        CARPETA_SALIDA / "fiabilidad_entropia_resumen_por_ventana_banda.csv",
        index=False,
    )

    df_status = pd.DataFrame(status_rows)
    if not df_status.empty:
        df_status.to_csv(
            CARPETA_SALIDA / "incidencias_entropia_split_half.csv", index=False
        )

    config_lines = [
        "Fiabilidad de entropía espectral de grafos EEG con wPLI",
        f"RUTA_BASE={RUTA_BASE}",
        f"ARCHIVO_LABELS={ARCHIVO_LABELS}",
        f"Ventanas_s={VENTANAS_SEGUNDOS}",
        f"Bandas={BANDAS}",
        f"MT_BANDWIDTH_HZ={MT_BANDWIDTH_HZ}",
        f"N_SPLIT_REPEATS={N_SPLIT_REPEATS}",
        f"MIN_EPOCAS_POR_MITAD={MIN_EPOCAS_POR_MITAD}",
        f"N_registros_incluidos={len(cohort_codes)}",
        "Cohorte común: al menos 2*MIN_EPOCAS_POR_MITAD épocas en todas las ventanas.",
        "Entropía: L=D-W; p_i=lambda_i/sum(lambda); H=-sum(p_i ln p_i); H_norm=H/ln(N).",
        "wPLI se estima por banda con spectral_connectivity_epochs, mode=multitaper, faverage=True.",
        "La copia filtrada 1-45 Hz se usa solo para ajustar ICA; wPLI se estima sobre la señal limpiada sin ese FIR.",
        "Las mitades se generan mediante partición aleatoria de épocas a lo largo del registro; no son dos mitades cronológicas contiguas.",
        "No se calculan p-valores ingenuos por la dependencia de registros del mismo sujeto.",
    ]
    (CARPETA_SALIDA / "configuracion_entropia.txt").write_text(
        "\n".join(config_lines), encoding="utf-8"
    )

    # Gráficas de resumen por banda y duración.
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for band in BAND_NAMES:
            d = df_summary[df_summary["banda"] == band].sort_values("ventana_s")
            y_col = "pearson_r_media_entre_splits"
            ax.plot(d["ventana_s"], d[y_col], marker="o", label=band)
        ax.set_xlabel("Duración de ventana (s)")
        ax.set_ylabel("Pearson r entre H de mitad A y mitad B")
        ax.set_title("Reproducibilidad relativa de la entropía espectral")
        ax.grid(alpha=0.3)
        ax.legend(title="Banda")
        fig.tight_layout()
        fig.savefig(CARPETA_SALIDA / "fiabilidad_entropia_correlacion.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for band in BAND_NAMES:
            d = df_summary[df_summary["banda"] == band].sort_values("ventana_s")
            y_col = "mae_H_norm_media_entre_splits"
            ax.plot(d["ventana_s"], d[y_col], marker="o", label=band)
        ax.set_xlabel("Duración de ventana (s)")
        ax.set_ylabel("MAE de entropía espectral normalizada")
        ax.set_title("Error absoluto split-half de la entropía espectral")
        ax.grid(alpha=0.3)
        ax.legend(title="Banda")
        fig.tight_layout()
        fig.savefig(CARPETA_SALIDA / "fiabilidad_entropia_error.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for band in BAND_NAMES:
            d = df_summary[df_summary["banda"] == band].sort_values("ventana_s")
            y_col = "icc_A1_acuerdo_absoluto_media_entre_splits"
            ax.plot(d["ventana_s"], d[y_col], marker="o", label=band)
        ax.axhline(0.0, linewidth=1, linestyle="--")
        ax.set_xlabel("Duración de ventana (s)")
        ax.set_ylabel("ICC(A,1), acuerdo absoluto")
        ax.set_title("Acuerdo absoluto de entropía espectral entre mitades")
        ax.grid(alpha=0.3)
        ax.legend(title="Banda")
        fig.tight_layout()
        fig.savefig(CARPETA_SALIDA / "fiabilidad_entropia_icc.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"AVISO: no se pudieron crear las gráficas: {exc}")

    elapsed = time.perf_counter() - start
    print("\n" + "=" * 80)
    print("ANÁLISIS DE FIABILIDAD DE ENTROPÍA TERMINADO")
    print("=" * 80)
    print(f"Registros incluidos en la cohorte común: {len(cohort_codes)}")
    print(f"Resultados guardados en: {CARPETA_SALIDA}")
    print(f"Tiempo total: {elapsed:.1f} s")
    if not df_summary.empty:
        cols = [
            "ventana_s", "banda",
            "pearson_r_media_entre_splits",
            "spearman_rho_media_entre_splits",
            "mae_H_norm_media_entre_splits",
            "rmse_H_norm_media_entre_splits",
            "icc_A1_acuerdo_absoluto_media_entre_splits",
            "n_registros_media_entre_splits",
        ]
        print("\nResumen (promedio entre particiones repetidas):")
        print(df_summary[cols].to_string(index=False))

    if not df_status.empty:
        print(f"\nIncidencias/exclusiones registradas: {len(df_status)}")
        print(f"Ver: {CARPETA_SALIDA / 'incidencias_entropia_split_half.csv'}")


if __name__ == "__main__":
    main()
