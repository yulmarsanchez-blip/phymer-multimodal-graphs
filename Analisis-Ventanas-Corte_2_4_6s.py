from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

from scipy.stats import kurtosis, wilcoxon, rankdata


# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_BASE = Path(
    r"C:\Users\Maria\Downloads\PhyMER-20250109T224509Z-001"
    r"\PhyMER\PhyMER Dataset\PhyMER Dataset\Emotiv"
)

archivo_labels = Path(
    r"C:\Users\Maria\Downloads\PhyMER-20250109T224509Z-001"
    r"\PhyMER\PhyMER Dataset\PhyMER Dataset\labels.csv"
)

CARPETA_SALIDA = Path(
    r"C:\Users\Maria\Downloads\Code-TG\Señal EEG"
)

CONFIGURACION = "analisis_cortes_2_4_6s"


# ============================================================
# SUJETOS Y VIDEOS
# ============================================================

# Para el análisis inferencial, conviene usar todos los sujetos disponibles.
# Para una prueba rápida del código, puede sustituirse por SUJETOS = [1].
SUJETOS = list(range(1, 31))

VIDEOS = list(range(1, 26))


# ============================================================
# PARÁMETROS
# ============================================================

# La duración de época se mantiene fija; así se comparan cortes distintos
# sin modificar la resolución temporal de las métricas por época.
DURACION_EPOCA_S = 2.0

# Duraciones de los segmentos de borde que se van a evaluar.
# Deben ser múltiplos enteros de DURACION_EPOCA_S.
DURACIONES_CORTE_S = (2.0, 4.0, 6.0)

# Número mínimo de épocas que deben permanecer en el interior del vídeo
# después de retirar conceptualmente ambos segmentos de borde.
MIN_EPOCAS_INTERIOR = 3

FILTRO_ICA_BAJO = 1.0
FILTRO_ICA_ALTO = 45.0

# Umbral de anomalía robusta
Z_ROBUSTO_UMBRAL = 3.5

# Nivel de significancia
ALPHA = 0.05

# Para considerar un efecto estadísticamente relevante:
# mínimo número de métricas significativas
MIN_METRICAS_SIGNIFICATIVAS = 2

# Umbral de tamaño del efecto de Wilcoxon
EFECTO_MINIMO = 0.5

# Proporción mínima de videos con anomalía
CONSISTENCIA_MINIMA = 2 / 3


# ============================================================
# MÉTRICAS
# ============================================================

METRICAS_TEMPORALES = [
    "rms",
    "std",
    "ptp",
    "kurtosis"
]

METRICAS_ESPECTRALES = [
    "slow_power_1_4",
    "high_power_30_45"
]

TODAS_METRICAS = (
    METRICAS_TEMPORALES
    + METRICAS_ESPECTRALES
)


# ============================================================
# FUNCIONES
# ============================================================

def extraer_codigo(path):

    patron = re.compile(
        r"(SUB\d+VID\d+)",
        re.IGNORECASE
    )

    m = patron.search(path.stem)

    if m:
        return m.group(1).upper()

    return None


def localizar_archivo(sujeto, video):

    codigo = (
        f"SUB{sujeto:02d}"
        f"VID{video:02d}"
    )

    # Usamos rglob para tolerar nombres como:
    #
    # SUB01VID01_eeg.set
    # SUB01VID01_eeg(1).set

    candidatos = list(
        RUTA_BASE.rglob(
            f"{codigo}*_eeg*.set"
        )
    )

    if not candidatos:
        return None

    return candidatos[0]


def calcular_metricas_temporales(datos):
    """
    datos:
        [n_canales, n_muestras]
    """

    rms = np.sqrt(
        np.mean(datos ** 2, axis=1)
    )

    std = np.std(
        datos,
        axis=1,
        ddof=1
    )

    ptp = np.ptp(
        datos,
        axis=1
    )

    kurt = kurtosis(
        datos,
        axis=1,
        fisher=True,
        bias=False,
        nan_policy="omit"
    )

    return {
        "rms": rms,
        "std": std,
        "ptp": ptp,
        "kurtosis": kurt
    }


def calcular_psd_bandas(epochs):

    """
    PSD sobre épocas de duración DURACION_EPOCA_S, mantenidas constantes
    al comparar segmentos de borde de distinta duración.

    No se aplica ningún nuevo filtro temporal.
    """

    sfreq = epochs.info["sfreq"]

    n_times = len(
        epochs.times
    )

    # Welch sobre toda la época
    n_fft = n_times
    n_per_seg = n_times

    spectrum = epochs.compute_psd(
        method="welch",
        fmin=1.0,
        fmax=45.0,
        n_fft=n_fft,
        n_per_seg=n_per_seg,
        n_overlap=0,
        picks="eeg",
        remove_dc=True,
        verbose=False
    )

    psd = spectrum.get_data()

    freqs = spectrum.freqs

    # psd:
    #
    # [n_epocas, n_canales, n_frecuencias]

    mascara_slow = (
        (freqs >= 1.0)
        &
        (freqs <= 4.0)
    )

    mascara_high = (
        (freqs >= 30.0)
        &
        (freqs <= 45.0)
    )

    slow_power = np.mean(
        psd[:, :, mascara_slow],
        axis=2
    )

    high_power = np.mean(
        psd[:, :, mascara_high],
        axis=2
    )

    # Promedio sobre canales
    slow_power = np.mean(
        slow_power,
        axis=1
    )

    high_power = np.mean(
        high_power,
        axis=1
    )

    return (
        slow_power,
        high_power
    )


def robust_zscore(valor, referencia):

    mediana = np.median(
        referencia
    )

    mad = np.median(
        np.abs(
            referencia - mediana
        )
    )

    if mad < 1e-12:

        if abs(valor - mediana) < 1e-12:
            return 0.0

        return np.inf

    return (
        0.6745
        * (valor - mediana)
        / mad
    )


def rank_biserial_effect(diferencias):

    """
    Tamaño de efecto rank-biserial para Wilcoxon.
    """

    diferencias = np.asarray(
        diferencias,
        dtype=float
    )

    diferencias = diferencias[
        np.isfinite(diferencias)
    ]

    diferencias = diferencias[
        diferencias != 0
    ]

    if len(diferencias) == 0:
        return np.nan

    rangos = rankdata(
        np.abs(diferencias)
    )

    suma_total = np.sum(
        rangos
    )

    suma_positivos = np.sum(
        rangos[diferencias > 0]
    )

    suma_negativos = np.sum(
        rangos[diferencias < 0]
    )

    efecto = (
        suma_positivos
        - suma_negativos
    ) / suma_total

    return efecto


def bh_fdr(pvalues):

    """
    Corrección Benjamini-Hochberg.
    """

    pvalues = np.asarray(
        pvalues,
        dtype=float
    )

    qvalues = np.full(
        len(pvalues),
        np.nan
    )

    validos = np.isfinite(
        pvalues
    )

    p = pvalues[validos]

    if len(p) == 0:
        return qvalues

    orden = np.argsort(p)

    p_ordenados = p[orden]

    n = len(p)

    q_ordenados = (
        p_ordenados * n
        / np.arange(1, n + 1)
    )

    q_ordenados = np.minimum.accumulate(
        q_ordenados[::-1]
    )[::-1]

    q_ordenados = np.clip(
        q_ordenados,
        0,
        1
    )

    q = np.empty_like(
        p_ordenados
    )

    q[orden] = q_ordenados

    qvalues[validos] = q

    return qvalues


# ============================================================
# PROCESAR UN VIDEO
# ============================================================

def procesar_video(
    sujeto,
    video,
    labels
):

    codigo = (
        f"SUB{sujeto:02d}"
        f"VID{video:02d}"
    )

    ruta = localizar_archivo(
        sujeto,
        video
    )

    if ruta is None:

        print(
            f"No encontrado: {codigo}"
        )

        return None

    print("\n" + "=" * 70)
    print(f"Procesando {codigo}")
    print("=" * 70)

    # --------------------------------------------------------
    # Cargar
    # --------------------------------------------------------

    raw = mne.io.read_raw_eeglab(
        str(ruta),
        preload=True,
        verbose=False
    )

    raw.pick("eeg")

    print(
        f"Canales: {len(raw.ch_names)}"
    )

    print(
        f"Frecuencia: "
        f"{raw.info['sfreq']:.2f} Hz"
    )

    print(
        f"Duración: "
        f"{raw.times[-1]:.2f} s"
    )

    # --------------------------------------------------------
    # ICA
    # --------------------------------------------------------

    raw_ica = raw.copy()

    raw_ica.filter(
        l_freq=FILTRO_ICA_BAJO,
        h_freq=FILTRO_ICA_ALTO,
        fir_design="firwin",
        verbose=False
    )

    ica = mne.preprocessing.ICA(
        n_components=len(raw.ch_names),
        random_state=42,
        max_iter="auto"
    )

    ica.fit(
        raw_ica,
        verbose=False
    )

    # --------------------------------------------------------
    # EOG proxy AF3
    # --------------------------------------------------------

    componentes_af3, _ = (
        ica.find_bads_eog(
            raw_ica,
            ch_name="AF3",
            threshold=3.0,
            verbose=False
        )
    )

    # --------------------------------------------------------
    # EOG proxy AF4
    # --------------------------------------------------------

    componentes_af4, _ = (
        ica.find_bads_eog(
            raw_ica,
            ch_name="AF4",
            threshold=3.0,
            verbose=False
        )
    )

    componentes_eog = sorted(
        set(componentes_af3)
        |
        set(componentes_af4)
    )

    ica.exclude = componentes_eog

    print(
        f"Componentes ICA excluidos: "
        f"{ica.exclude}"
    )

    # --------------------------------------------------------
    # Aplicar ICA SOBRE RAW NO FILTRADO
    # --------------------------------------------------------

    raw_limpio = raw.copy()

    ica.apply(
        raw_limpio,
        verbose=False
    )

    # --------------------------------------------------------
    # Re-referencia
    # --------------------------------------------------------

    raw_limpio.set_eeg_reference(
        ref_channels="average",
        projection=False,
        verbose=False
    )

    # --------------------------------------------------------
    # EPOCADO FIJO (NO CAMBIA ENTRE TAMAÑOS DE CORTE)
    # --------------------------------------------------------

    epochs = mne.make_fixed_length_epochs(
        raw_limpio,
        duration=DURACION_EPOCA_S,
        preload=True,
        reject_by_annotation=True,
        verbose=False
    )

    n_epocas = len(epochs)

    print(
        f"Épocas de {DURACION_EPOCA_S:g} s: {n_epocas}"
    )

    if n_epocas < MIN_EPOCAS_INTERIOR + 2:
        print(
            "Muy pocas épocas para evaluar de forma fiable "
            "los segmentos de borde y el interior."
        )
        return None

    # --------------------------------------------------------
    # DATOS
    # --------------------------------------------------------

    datos = epochs.get_data()

    # [épocas, canales, tiempo]

    n_epocas = datos.shape[0]

    resultados = []

    # --------------------------------------------------------
    # MÉTRICAS TEMPORALES
    # --------------------------------------------------------

    for epoca in range(
        n_epocas
    ):

        datos_epoca = datos[
            epoca
        ]

        metricas = (
            calcular_metricas_temporales(
                datos_epoca
            )
        )

        fila = {
            "codigo": codigo,
            "sujeto": sujeto,
            "video": video,
            "epoca": epoca + 1,
            "duracion_epoca_s": DURACION_EPOCA_S,
            "inicio_s": epoca * DURACION_EPOCA_S,
            "final_s": (
                (epoca + 1)
                * DURACION_EPOCA_S
            )
        }

        for nombre, valores in (
            metricas.items()
        ):

            # Promedio robusto sobre canales
            fila[nombre] = np.median(
                valores
            )

        resultados.append(
            fila
        )

    df = pd.DataFrame(
        resultados
    )

    # --------------------------------------------------------
    # PSD
    # --------------------------------------------------------

    slow_power, high_power = (
        calcular_psd_bandas(
            epochs
        )
    )

    df["slow_power_1_4"] = (
        slow_power
    )

    df["high_power_30_45"] = (
        high_power
    )

    # --------------------------------------------------------
    # COMPARAR SEGMENTOS DE BORDE DE 2, 4 Y 6 S CON EL INTERIOR
    # --------------------------------------------------------

    resumen_video = []
    for corte_s in DURACIONES_CORTE_S:
        n_borde_float = corte_s / DURACION_EPOCA_S
        n_borde = int(round(n_borde_float))

        # No permitir cortes que no coincidan con un número entero de épocas.
        if not np.isclose(n_borde_float, n_borde, atol=1e-9):
            raise ValueError(
                f"El corte de {corte_s:g} s no es múltiplo entero de "
                f"DURACION_EPOCA_S={DURACION_EPOCA_S:g} s."
            )

        # Épocas que componen los segmentos inicial y final.
        indices_inicio = np.arange(0, n_borde)
        indices_final = np.arange(n_epocas - n_borde, n_epocas)
        indices_interior = np.arange(n_borde, n_epocas - n_borde)

        print(
            f"  Corte {corte_s:g} s: {n_borde} épocas por borde; "
            f"{len(indices_interior)} épocas interiores."
        )

        for lado, indices_borde in (
            ("inicio", indices_inicio),
            ("final", indices_final),
        ):
            fila = {
                "codigo": codigo,
                "sujeto": sujeto,
                "video": video,
                "duracion_epoca_s": DURACION_EPOCA_S,
                "corte_s": float(corte_s),
                "n_epocas_total": n_epocas,
                "n_epocas_borde": len(indices_borde),
                "n_epocas_interior": len(indices_interior),
                "segundos_borde": len(indices_borde) * DURACION_EPOCA_S,
                "segundos_interior": len(indices_interior) * DURACION_EPOCA_S,
                "lado": lado,
            }

            if len(indices_interior) < MIN_EPOCAS_INTERIOR:
                fila["estado"] = "INSUFICIENTES_EPOCAS_INTERIOR"
                for metrica in TODAS_METRICAS:
                    fila[f"{metrica}_borde"] = np.nan
                    fila[f"{metrica}_interior_mediana"] = np.nan
                    fila[f"{metrica}_diferencia"] = np.nan
                    fila[f"{metrica}_z_robusto"] = np.nan
                    fila[f"{metrica}_anomala"] = False
                    fila[f"{metrica}_fraccion_epocas_borde_anomala"] = np.nan
                resumen_video.append(fila)
                continue

            fila["estado"] = "OK"

            for metrica in TODAS_METRICAS:
                valores_borde = df.loc[
                    indices_borde, metrica
                ].to_numpy(dtype=float)

                interior = df.loc[
                    indices_interior, metrica
                ].to_numpy(dtype=float)

                # Resumen robusto del segmento de borde: mediana de las
                # épocas que ocupan los primeros/últimos corte_s segundos.
                borde = float(np.nanmedian(valores_borde))
                mediana_int = float(np.nanmedian(interior))
                diferencia = borde - mediana_int
                z = robust_zscore(borde, interior)

                fila[f"{metrica}_borde"] = borde
                fila[f"{metrica}_interior_mediana"] = mediana_int
                fila[f"{metrica}_diferencia"] = diferencia
                fila[f"{metrica}_z_robusto"] = z
                # Si la dispersión robusta interior es cero, robust_zscore
                # puede devolver +/-inf para una desviación no nula; eso sí
                # debe contar como anomalía. NaN, en cambio, no es evaluable.
                fila[f"{metrica}_anomala"] = bool(abs(z) > Z_ROBUSTO_UMBRAL)

                # Complemento útil para cortes de 4/6 s: la mediana del
                # segmento puede ocultar una anomalía aislada. Registramos
                # qué fracción de las épocas del borde supera el umbral,
                # evaluada contra la distribución de las épocas interiores.
                z_epocas_borde = np.array([
                    robust_zscore(float(valor), interior)
                    for valor in valores_borde
                ], dtype=float)
                z_evaluable = z_epocas_borde[~np.isnan(z_epocas_borde)]
                fila[f"{metrica}_fraccion_epocas_borde_anomala"] = (
                    float(np.mean(np.abs(z_evaluable) > Z_ROBUSTO_UMBRAL))
                    if len(z_evaluable) else np.nan
                )

            resumen_video.append(fila)

    return df, pd.DataFrame(resumen_video)


# ============================================================
# MAIN
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning
    )

    CARPETA_SALIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    carpeta = (
        CARPETA_SALIDA
        / CONFIGURACION
    )

    carpeta.mkdir(
        parents=True,
        exist_ok=True
    )

    labels = pd.read_csv(
        archivo_labels
    )

    todos_epoch = []
    todos_resumen = []

    # --------------------------------------------------------
    # RECORRER
    # --------------------------------------------------------

    for sujeto in SUJETOS:

        for video in VIDEOS:

            resultado = (
                procesar_video(
                    sujeto,
                    video,
                    labels
                )
            )

            if resultado is None:
                continue

            df_epoch, df_resumen = (
                resultado
            )

            todos_epoch.append(
                df_epoch
            )

            todos_resumen.append(
                df_resumen
            )

            # Guardar análisis individual

            codigo = (
                f"SUB{sujeto:02d}"
                f"VID{video:02d}"
            )

            df_epoch.to_csv(
                carpeta
                / f"{codigo}_metricas_epocas.csv",
                index=False
            )

            df_resumen.to_csv(
                carpeta
                / f"{codigo}_borde_vs_interior_por_corte.csv",
                index=False
            )

    if not todos_resumen:

        print(
            "\nNo se obtuvieron resultados."
        )

        return

    # ========================================================
    # UNIR RESULTADOS
    # ========================================================

    df_resumen = pd.concat(
        todos_resumen,
        ignore_index=True
    )

    df_epoch = pd.concat(
        todos_epoch,
        ignore_index=True
    )

    df_resumen.to_csv(
        carpeta
        / "todos_borde_vs_interior_por_corte.csv",
        index=False
    )

    df_epoch.to_csv(
        carpeta
        / "todas_metricas_epocas.csv",
        index=False
    )

    # ========================================================
    # ANÁLISIS ESTADÍSTICO POR DURACIÓN DE CORTE
    # ========================================================

    resultados_estadisticos = []

    for corte_s in DURACIONES_CORTE_S:
        for lado in ("inicio", "final"):
            datos_lado = df_resumen[
                (df_resumen["corte_s"] == float(corte_s))
                & (df_resumen["lado"] == lado)
                & (df_resumen["estado"] == "OK")
            ].copy()

            for metrica in TODAS_METRICAS:
                columna = f"{metrica}_diferencia"
                z_columna = f"{metrica}_z_robusto"

                datos_validos = datos_lado[
                    np.isfinite(datos_lado[columna])
                ].copy()

                # Los vídeos de un mismo sujeto no son independientes.
                # Para la inferencia, consolidamos las diferencias dentro
                # de cada sujeto usando la mediana de sus vídeos.
                diferencias_sujeto = (
                    datos_validos.groupby("sujeto")[columna]
                    .median()
                    .dropna()
                    .to_numpy(dtype=float)
                )

                if len(diferencias_sujeto) < 3:
                    p = np.nan
                    efecto = np.nan
                else:
                    try:
                        prueba = wilcoxon(
                            diferencias_sujeto,
                            alternative="two-sided",
                            method="auto",
                        )
                        p = float(prueba.pvalue)
                    except ValueError:
                        p = np.nan

                    efecto = rank_biserial_effect(diferencias_sujeto)

                z = datos_validos[z_columna].to_numpy(dtype=float)
                z = z[~np.isnan(z)]

                # Consistencia descriptiva: primero se calcula la fracción
                # de vídeos anómalos de cada sujeto y luego se promedian
                # los sujetos, para no dar más peso a quien aporta más vídeos.
                por_sujeto_anomalia = datos_validos.groupby("sujeto")[z_columna].apply(
                    lambda x: np.mean(
                        np.abs(x.to_numpy(dtype=float)[~np.isnan(x.to_numpy(dtype=float))])
                        > Z_ROBUSTO_UMBRAL
                    )
                    if (~np.isnan(x.to_numpy(dtype=float))).any() else np.nan
                ).dropna()

                fraccion_anomala = (
                    float(por_sujeto_anomalia.mean())
                    if len(por_sujeto_anomalia) else np.nan
                )
                mediana_abs_z = float(np.median(np.abs(z))) if len(z) else np.nan

                resultados_estadisticos.append({
                    "corte_s": float(corte_s),
                    "lado": lado,
                    "metrica": metrica,
                    "n_videos": int(datos_validos["codigo"].nunique()),
                    "n_sujetos": int(len(diferencias_sujeto)),
                    "diferencia_mediana": float(np.median(diferencias_sujeto)) if len(diferencias_sujeto) else np.nan,
                    "diferencia_media": float(np.mean(diferencias_sujeto)) if len(diferencias_sujeto) else np.nan,
                    "p_valor": p,
                    "efecto_rank_biserial": efecto,
                    "fraccion_videos_anomala_promedio_sujeto": fraccion_anomala,
                    "mediana_abs_z": mediana_abs_z,
                })

    df_stats = pd.DataFrame(resultados_estadisticos)

    if df_stats.empty:
        print("No se obtuvieron suficientes datos para la inferencia estadística.")
        return

    # FDR global para todas las comparaciones definidas: tres cortes,
    # dos lados y seis métricas.
    df_stats["q_valor_FDR"] = bh_fdr(df_stats["p_valor"].to_numpy())
    df_stats["significativo_FDR"] = df_stats["q_valor_FDR"] < ALPHA
    df_stats["efecto_relevante"] = (
        np.abs(df_stats["efecto_rank_biserial"]) >= EFECTO_MINIMO
    )
    df_stats["consistente"] = (
        df_stats["fraccion_videos_anomala_promedio_sujeto"] >= CONSISTENCIA_MINIMA
    )

    # Decisión descriptiva para cada tamaño de corte y lado.
    for corte_s in DURACIONES_CORTE_S:
        for lado in ("inicio", "final"):
            mascara = (
                (df_stats["corte_s"] == float(corte_s))
                & (df_stats["lado"] == lado)
            )
            tabla = df_stats.loc[mascara]
            n_metricas = int((tabla["significativo_FDR"] & tabla["efecto_relevante"]).sum())
            n_consistentes = int(tabla["consistente"].sum())

            if (
                n_metricas >= MIN_METRICAS_SIGNIFICATIVAS
                and n_consistentes >= MIN_METRICAS_SIGNIFICATIVAS
            ):
                decision = "EVIDENCIA_A_FAVOR_DE_CORTE"
            elif n_metricas > 0:
                decision = "EVIDENCIA_INSUFICIENTE"
            else:
                decision = "NO_EVIDENCIA_DE_CORTE"

            df_stats.loc[mascara, "decision_global"] = decision
            print("\n" + "=" * 70)
            print(f"CORTE: {corte_s:g} s | BORDE: {lado.upper()}")
            print(f"Métricas con evidencia (FDR + efecto): {n_metricas}")
            print(f"Métricas con anomalía consistente: {n_consistentes}")
            print(f"Decisión según criterio predefinido: {decision}")
            print("=" * 70)

    # ========================================================
    # GUARDAR RESULTADOS ESTADÍSTICOS
    # ========================================================

    df_stats.to_csv(carpeta / "estadistica_borde_vs_interior_por_corte.csv", index=False)

    columnas_mostrar = [
        "corte_s", "lado", "metrica", "n_videos", "n_sujetos",
        "diferencia_mediana", "p_valor", "q_valor_FDR",
        "efecto_rank_biserial", "fraccion_videos_anomala_promedio_sujeto",
        "mediana_abs_z", "decision_global",
    ]
    print("\nRESULTADOS ESTADÍSTICOS POR CORTE")
    print(df_stats[columnas_mostrar].to_string(index=False))

    # ========================================================
    # GRÁFICAS: ANOMALÍAS ROBUSTAS POR CORTE Y LADO
    # ========================================================

    fig, axes = plt.subplots(
        nrows=2, ncols=len(DURACIONES_CORTE_S),
        figsize=(5.5 * len(DURACIONES_CORTE_S), 8),
        squeeze=False,
    )
    metricas_grafica = ["rms", "std", "ptp", "high_power_30_45"]

    for row_idx, lado in enumerate(("inicio", "final")):
        for col_idx, corte_s in enumerate(DURACIONES_CORTE_S):
            ax = axes[row_idx, col_idx]
            datos_plot = df_resumen[
                (df_resumen["lado"] == lado)
                & (df_resumen["corte_s"] == float(corte_s))
                & (df_resumen["estado"] == "OK")
            ].copy()

            # Los puntos se muestran en el orden sujeto-vídeo; cada punto
            # es un resumen por vídeo, no una época independiente.
            datos_plot = datos_plot.sort_values(["sujeto", "video"])
            posiciones = np.arange(len(datos_plot))
            for metrica in metricas_grafica:
                z_abs = np.abs(datos_plot[f"{metrica}_z_robusto"].to_numpy(dtype=float))
                ax.plot(posiciones, z_abs, marker="o", markersize=2.5, linewidth=0.8, label=metrica)

            ax.axhline(
                Z_ROBUSTO_UMBRAL,
                linestyle="--",
                linewidth=1.2,
                label=f"umbral z = {Z_ROBUSTO_UMBRAL:g}",
            )
            ax.set_title(f"Borde {lado} — corte {corte_s:g} s")
            ax.set_xlabel("Registro ordenado por sujeto y vídeo")
            ax.set_ylabel("|z robusto|")
            ax.grid(alpha=0.3)
            if row_idx == 0 and col_idx == len(DURACIONES_CORTE_S) - 1:
                ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Comparación de segmentos de borde e interior; épocas de {DURACION_EPOCA_S:g} s",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(
        carpeta / "anomalia_borde_vs_interior_por_corte.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nResultados guardados en:")
    print(carpeta)


# ============================================================
# EJECUTAR
# ============================================================

if __name__ == "__main__":
    main()