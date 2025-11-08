import math
import time
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Configuration & Constants
# -----------------------------
st.set_page_config(
    page_title="Wind Turbine: Protection & Performance",
    page_icon="??",
    layout="wide",
)

RATED_POWER_KW = 3000.0  # 3 MW
CUT_IN_MS = 3.0
RATED_WS_MS = 12.0
CUT_OUT_MS = 25.0
ROTOR_DIAMETER_M = 120.0  # example utility-scale rotor
SWEPT_AREA_M2 = math.pi * (ROTOR_DIAMETER_M / 2.0) ** 2

# -----------------------------
# Utility functions
# -----------------------------

def compute_power_curve(wind_speed_ms: np.ndarray, air_density: float) -> np.ndarray:
    """Idealized power curve: region II cubic to rated, then flat till cut-out.
    Returns power in kW.
    """
    v = wind_speed_ms
    power = np.zeros_like(v)

    # Region I: below cut-in
    mask_region1 = v < CUT_IN_MS
    power[mask_region1] = 0.0

    # Region II: cubic ramp-up
    mask_region2 = (v >= CUT_IN_MS) & (v < RATED_WS_MS)
    # Normalize cubic between cut-in and rated
    v_norm = (v[mask_region2] - CUT_IN_MS) / (RATED_WS_MS - CUT_IN_MS)
    power[mask_region2] = (RATED_POWER_KW * (v_norm ** 3)) * (air_density / 1.225)

    # Region III: rated power flat until cut-out
    mask_region3 = (v >= RATED_WS_MS) & (v < CUT_OUT_MS)
    power[mask_region3] = RATED_POWER_KW * (air_density / 1.225)

    # Region IV: cut-out, shutdown
    mask_region4 = v >= CUT_OUT_MS
    power[mask_region4] = 0.0

    return power


def simulate_turbine_data(
    num_samples: int,
    mean_wind: float,
    std_wind: float,
    turbulence_intensity: float,
    air_density: float,
    ambient_temp_c: float,
    failure_rate: float,
    noise_level: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate SCADA-like time series and inject some failure modes.
    failure_rate in [0, 1] is the probability per sample of a failure state.
    """
    rng = np.random.default_rng(seed)

    # Base wind speed with turbulence
    base_ws = rng.normal(mean_wind, std_wind, size=num_samples)
    turb = rng.normal(0.0, turbulence_intensity * np.maximum(base_ws, 0.0))
    wind_speed = np.clip(base_ws + turb, 0.0, None)

    yaw_error_deg = rng.normal(0.0, 6.0, size=num_samples)  # +/- few degrees
    pitch_deg = np.clip(rng.normal(2.0, 1.0, size=num_samples), 0.0, 15.0)

    # Ideal power + yaw/pitch losses
    ideal_power_kw = compute_power_curve(wind_speed, air_density)
    yaw_loss = np.cos(np.deg2rad(np.abs(yaw_error_deg)))
    pitch_loss = 1.0 - 0.01 * np.maximum(0.0, pitch_deg - 2.0)
    power_kw = ideal_power_kw * yaw_loss * np.clip(pitch_loss, 0.9, 1.0)

    # Add measurement noise
    power_kw = np.clip(power_kw + rng.normal(0, noise_level * RATED_POWER_KW, size=num_samples), 0.0, None)

    # Rotor speed proxy (not physically exact)
    rotor_speed_rpm = np.clip(6.0 * wind_speed + rng.normal(0, 2.0, size=num_samples), 0.0, 20.0) * 10.0

    # Temperatures and vibration (baseline)
    generator_temp_c = ambient_temp_c + 20.0 + 0.0005 * power_kw + rng.normal(0, 1.5, size=num_samples)
    gearbox_temp_c = ambient_temp_c + 15.0 + 0.0004 * power_kw + rng.normal(0, 1.8, size=num_samples)
    vibration_mm_s = np.clip(rng.normal(2.0, 0.6, size=num_samples) + 0.0005 * np.maximum(0.0, power_kw - 1500.0), 0.5, None)

    # Failures
    failure_label = np.zeros(num_samples, dtype=int)  # 0 normal
    protection_action = np.array(["NORMAL"] * num_samples)

    for i in range(num_samples):
        if rng.random() < failure_rate:
            # Randomly choose a failure type
            f = rng.integers(1, 6)  # 1..5
            failure_label[i] = f
            if f == 1:
                # Overspeed
                rotor_speed_rpm[i] *= 1.5
                protection_action[i] = "SHUTDOWN"
            elif f == 2:
                # Overtemperature (generator)
                generator_temp_c[i] += rng.uniform(20, 40)
                protection_action[i] = "SHUTDOWN"
            elif f == 3:
                # Excessive vibration
                vibration_mm_s[i] += rng.uniform(5, 10)
                protection_action[i] = "SHUTDOWN"
            elif f == 4:
                # Grid fault -> power collapse
                power_kw[i] = np.clip(power_kw[i] * rng.uniform(0.0, 0.2), 0.0, None)
                protection_action[i] = "CURTAIL"
            elif f == 5:
                # Pitch fault -> yaw/pitch inefficiency
                pitch_deg[i] = np.clip(pitch_deg[i] + rng.uniform(5, 10), 0.0, 20.0)
                power_kw[i] = power_kw[i] * rng.uniform(0.6, 0.85)
                protection_action[i] = "CURTAIL"

    timestamps = pd.date_range("2024-01-01", periods=num_samples, freq="5min")

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "wind_speed_ms": wind_speed,
            "yaw_error_deg": yaw_error_deg,
            "pitch_deg": pitch_deg,
            "rotor_speed_rpm": rotor_speed_rpm,
            "power_kw": power_kw,
            "generator_temp_c": generator_temp_c,
            "gearbox_temp_c": gearbox_temp_c,
            "vibration_mm_s": vibration_mm_s,
            "ambient_temp_c": ambient_temp_c,
            "air_density": air_density,
            "failure_label": failure_label,
            "protection_action": protection_action,
        }
    )
    return df


# -----------------------------
# Simple ML (NumPy-based)
# -----------------------------

def build_polynomial_features(x: np.ndarray, degree: int = 3) -> np.ndarray:
    X = np.ones((x.shape[0], degree + 1))
    for d in range(1, degree + 1):
        X[:, d] = x ** d
    return X


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # theta = (X^T X)^-1 X^T y
    xtx = X.T @ X
    xty = X.T @ y
    theta = np.linalg.pinv(xtx) @ xty
    return theta


def predict_ols(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return X @ theta


def knn_predict(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, k: int = 5) -> np.ndarray:
    preds = []
    for x in test_X:
        dists = np.linalg.norm(train_X - x, axis=1)
        idx = np.argpartition(dists, kth=min(k, len(dists)-1))[:k]
        votes = train_y[idx]
        # majority vote
        values, counts = np.unique(votes, return_counts=True)
        preds.append(values[np.argmax(counts)])
    return np.array(preds)


def zscore(series: np.ndarray) -> np.ndarray:
    mu = np.mean(series)
    sigma = np.std(series) + 1e-9
    return (series - mu) / sigma


# -----------------------------
# Protection logic
# -----------------------------

def decide_protection_action(
    row: pd.Series,
    thresholds: Dict[str, float],
) -> str:
    if row["wind_speed_ms"] >= thresholds["cutout_ws_ms"]:
        return "SHUTDOWN"
    if row["rotor_speed_rpm"] >= thresholds["max_rotor_rpm"]:
        return "SHUTDOWN"
    if row["generator_temp_c"] >= thresholds["max_generator_c"]:
        return "SHUTDOWN"
    if row["vibration_mm_s"] >= thresholds["max_vibration_mm_s"]:
        return "SHUTDOWN"
    # Curtail conditions
    if abs(row["yaw_error_deg"]) >= thresholds["max_yaw_error_deg"]:
        return "CURTAIL"
    if row["ambient_temp_c"] >= thresholds["max_ambient_c"]:
        return "CURTAIL"
    return "NORMAL"


# -----------------------------
# Streamlit UI
# -----------------------------
with st.sidebar:
    st.header("Simulation Parameters")
    num_samples = st.slider("Samples", 200, 5000, 1200, step=100)

    mean_wind = st.slider("Mean wind speed [m/s]", 2.0, 14.0, 8.5, 0.1)
    std_wind = st.slider("Wind std [m/s]", 0.1, 4.0, 1.8, 0.1)
    turbulence = st.slider("Turbulence intensity factor", 0.0, 0.6, 0.1, 0.01)

    air_density = st.slider("Air density [kg/m?]", 1.0, 1.35, 1.225, 0.005)
    ambient_temp = st.slider("Ambient temp [?C]", -20.0, 45.0, 12.0, 0.5)

    failure_rate = st.slider("Failure probability per point", 0.0, 0.2, 0.03, 0.005)
    noise_level = st.slider("Measurement noise level (power)", 0.0, 0.1, 0.02, 0.005)

    seed = st.number_input("Random seed", 0, 1_000_000, 42, step=1)

    st.divider()
    st.header("Protection Thresholds")
    th_cutout = st.slider("Cut-out wind speed [m/s]", 20.0, 30.0, CUT_OUT_MS, 0.5)
    th_rotor = st.slider("Max rotor speed [rpm]", 80.0, 250.0, 180.0, 5.0)
    th_gen = st.slider("Max generator temp [?C]", 60.0, 140.0, 95.0, 1.0)
    th_vib = st.slider("Max vibration [mm/s]", 3.0, 25.0, 12.0, 0.5)
    th_yaw = st.slider("Max yaw error [deg]", 4.0, 30.0, 15.0, 1.0)
    th_amb = st.slider("Max ambient temp [?C]", -10.0, 50.0, 40.0, 1.0)

    thresholds = {
        "cutout_ws_ms": th_cutout,
        "max_rotor_rpm": th_rotor,
        "max_generator_c": th_gen,
        "max_vibration_mm_s": th_vib,
        "max_yaw_error_deg": th_yaw,
        "max_ambient_c": th_amb,
    }

st.title("?? Wind Turbine Failure Protection & Performance Analysis")

with st.expander("About this app", expanded=False):
    st.write(
        "This app simulates SCADA-like data, trains simple ML models (NumPy-based) for power curve and failure detection, and evaluates protection actions."
    )

# Generate data
with st.spinner("Simulating data..."):
    df = simulate_turbine_data(
        num_samples=num_samples,
        mean_wind=mean_wind,
        std_wind=std_wind,
        turbulence_intensity=turbulence,
        air_density=air_density,
        ambient_temp_c=ambient_temp,
        failure_rate=failure_rate,
        noise_level=noise_level,
        seed=int(seed),
    )

st.success(f"Generated {len(df):,} samples")

# Display preview
st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)

# Power curve visualization
st.subheader("Power Curve and Model Fit")
col_pc1, col_pc2 = st.columns([1, 1])
with col_pc1:
    st.caption("Scatter: Power vs Wind Speed (colored by failure)")
    st.scatter_chart(
        df,
        x="wind_speed_ms",
        y="power_kw",
        color="failure_label",
        size=None,
        height=320,
    )

# Train power model on normal data only
normal_df = df[df["failure_label"] == 0]
X_poly = build_polynomial_features(normal_df["wind_speed_ms"].to_numpy(), degree=3)
y = normal_df["power_kw"].to_numpy()
coef = fit_ols(X_poly, y)

# Predict for all
X_all = build_polynomial_features(df["wind_speed_ms"].to_numpy(), degree=3)
y_hat = predict_ols(X_all, coef)

with col_pc2:
    st.caption("Predicted vs Actual Power")
    compare_df = pd.DataFrame({
        "timestamp": df["timestamp"],
        "Actual Power [kW]": df["power_kw"],
        "Predicted Power [kW]": y_hat,
    }).set_index("timestamp")
    st.line_chart(compare_df, height=320)

# Residuals and anomaly score
residuals = df["power_kw"].to_numpy() - y_hat
residual_z = zscore(residuals)

# Simple binary failure classification with kNN
st.subheader("Failure Classification (kNN)")
features = [
    "wind_speed_ms",
    "yaw_error_deg",
    "pitch_deg",
    "rotor_speed_rpm",
    "generator_temp_c",
    "gearbox_temp_c",
    "vibration_mm_s",
]

# Prepare train/test split
mask_train = np.zeros(len(df), dtype=bool)
mask_train[: int(0.7 * len(df))] = True
np.random.default_rng(0).shuffle(mask_train)

train_X = df.loc[mask_train, features].to_numpy()
train_y = (df.loc[mask_train, "failure_label"].to_numpy() != 0).astype(int)  # 1 if failed

test_X = df.loc[~mask_train, features].to_numpy()
test_y = (df.loc[~mask_train, "failure_label"].to_numpy() != 0).astype(int)

knn_k = st.slider("k (neighbors)", 1, 15, 5, 1)
preds = knn_predict(train_X, train_y, test_X, k=int(knn_k))

acc = (preds == test_y).mean() if len(test_y) else float("nan")
st.metric("Test Accuracy (binary failure)", f"{acc*100:.1f}%")

# Protection decision on current point
st.subheader("Protection Decision Simulator")
row_index = st.slider("Select sample index", 0, len(df) - 1, len(df) - 1)
row = df.iloc[row_index]
action = decide_protection_action(row, thresholds)

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.metric("Wind [m/s]", f"{row['wind_speed_ms']:.1f}")
    st.metric("Rotor [rpm]", f"{row['rotor_speed_rpm']:.0f}")
with c2:
    st.metric("Gen Temp [?C]", f"{row['generator_temp_c']:.1f}")
    st.metric("Vibration [mm/s]", f"{row['vibration_mm_s']:.1f}")
with c3:
    st.metric("Yaw Err [deg]", f"{row['yaw_error_deg']:.1f}")
    st.metric("Decision", action)

# Aggregate KPIs
st.subheader("Performance KPIs")
interval_hours = 5.0 / 60.0  # 5 minutes
energy_mwh = df["power_kw"].sum() * interval_hours / 1000.0
capacity_factor = df["power_kw"].mean() / RATED_POWER_KW

k1, k2, k3 = st.columns([1, 1, 1])
with k1:
    st.metric("Energy [MWh]", f"{energy_mwh:.2f}")
with k2:
    st.metric("Capacity Factor", f"{capacity_factor*100:.1f}%")
with k3:
    st.metric("Failure Rate (observed)", f"{(df['failure_label']!=0).mean()*100:.1f}%")

# Residual anomaly chart
st.subheader("Power Residual Anomaly Score (z-score)")
res_df = pd.DataFrame({
    "timestamp": df["timestamp"],
    "Residual z-score": residual_z,
}).set_index("timestamp")
st.line_chart(res_df, height=260)

# Time-series quick views
st.subheader("Time Series Overview")
series_cols = st.multiselect(
    "Select signals",
    [
        "power_kw",
        "wind_speed_ms",
        "rotor_speed_rpm",
        "generator_temp_c",
        "gearbox_temp_c",
        "vibration_mm_s",
        "yaw_error_deg",
        "pitch_deg",
    ],
    default=["power_kw", "wind_speed_ms", "generator_temp_c"],
)
if series_cols:
    ts_df = df[["timestamp", *series_cols]].set_index("timestamp")
    st.line_chart(ts_df, height=280)

# Download
st.subheader("Download Data")
st.download_button(
    label="Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="turbine_simulation.csv",
    mime="text/csv",
)

st.caption(
    "Note: ML models implemented using NumPy for broad compatibility in browser (stlite)."
)
