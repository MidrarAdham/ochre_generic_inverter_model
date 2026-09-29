"""
Figures from the OCHRE + GridLAB-D co-simulation results in cosim/output/.

  1. cosim_volt_var_curve.png: battery Q vs. the voltage OCHRE received from GridLAB-D,
     on top of the IEEE 1547 Cat. B curve
  2. cosim_timeseries.png: swing and house voltage, battery Q, and battery P over the run
  3. cosim_exchange.png: what OCHRE published vs. what GridLAB-D solved with, zoomed in
     on the largest Q change to show the one-step exchange lag

Runs the co-simulation first if there are no results. Usage:
    python cosim/plot_cosim.py [output_dir]   (default: cosim/figures)
"""

import datetime as dt
import os
import sys

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import run_cosim
from ochre_federate import START, V_BASE  # also puts ../src on sys.path
from inverter import IEEE1547_CAT_B_VOLT_VAR
from demo import BAND, MUTED, SERIES, SURFACE, TEXT, TEXT_2, plt, set_style

HERE = os.path.dirname(os.path.abspath(__file__))
V_SWING_BASE = 7200  # substation nominal voltage, in V


def load_results():
    out = run_cosim.OUTPUT
    if not os.path.exists(os.path.join(out, "ochre_federate.csv")):
        codes = run_cosim.run()
        if any(codes.values()):
            raise RuntimeError(f"Co-simulation failed: {codes}, see logs in {out}")

    df = pd.read_csv(os.path.join(out, "ochre_federate.csv")).set_index("Time (s)")
    df.index = df.index.astype(int)
    gld_v = run_cosim.read_gld_recorder(os.path.join(out, "gld_voltage.csv"))
    gld_s = run_cosim.read_gld_recorder(os.path.join(out, "gld_load.csv"))
    df["House Voltage GLD (-)"] = np.abs(gld_v.loc[df.index].values) / V_BASE
    df["GLD Load P (kW)"] = gld_s.loc[df.index].values.real / 1000
    df["GLD Load Q (kVAR)"] = gld_s.loc[df.index].values.imag / 1000

    player = pd.read_csv(
        os.path.join(HERE, "swing_voltage.player"), header=None, names=["Time", "V"]
    )
    swing = pd.Series(
        player["V"].map(lambda v: abs(complex(v))).values / V_SWING_BASE,
        index=(pd.to_datetime(player["Time"]) - START).dt.total_seconds().astype(int),
    )
    df["Swing Voltage (-)"] = swing.reindex(df.index, method="ffill")

    df.index = pd.to_datetime(START) + pd.to_timedelta(df.index, unit="s")
    return df


def format_time_axis(ax, minutes=None):
    if minutes:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=minutes))
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Simulation time")


def mark_reactive_transitions(axes, df, start=None, end=None):
    """Mark battery Q state changes at logged timestamps (negative Q injects)."""
    q = df["Battery Reactive Power (kVAR)"]
    state = pd.Series(np.select([q < -1e-6, q > 1e-6], [-1, 1], default=0), index=df.index)
    previous = state.shift()
    changes = state.ne(previous)
    start = df.index[0] if start is None else start
    end = df.index[-1] if end is None else end
    for timestamp in state.index[changes]:
        if not start <= timestamp <= end:
            continue
        current, prior = state.loc[timestamp], previous.loc[timestamp]
        if pd.isna(prior):
            if current == 0:
                continue
            label = "Injecting at start" if current < 0 else "Absorbing at start"
        elif current == 0:
            label = "Injection stops" if prior < 0 else "Absorption stops"
        elif current < 0:
            label = "Absorption → injection" if prior > 0 else "Injection starts"
        else:
            label = "Injection → absorption" if prior < 0 else "Absorption starts"
        color = SERIES[0] if current < 0 else SERIES[1] if current > 0 else TEXT_2
        for ax in axes:
            ax.axvline(timestamp, color=color, ls="--", lw=1, alpha=0.75)
        axes[-1].annotate(
            f"{label} {timestamp:%H:%M:%S}",
            xy=(timestamp, 0.04), xycoords=("data", "axes fraction"),
            xytext=(4, 0), textcoords="offset points", rotation=90,
            ha="left", va="bottom", fontsize=7, color=color,
            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=2),
        )


def plot_volt_var_curve(df):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    voltages, q_pu = IEEE1547_CAT_B_VOLT_VAR
    v_curve = np.linspace(0.88, 1.12, 481)
    ax.axvspan(voltages[1], voltages[2], color=BAND, zorder=0)
    ax.plot(
        v_curve, np.interp(v_curve, voltages, q_pu),
        color=TEXT_2, lw=1.2, ls="--", label="IEEE 1547 Cat. B curve",
    )
    q = df["Battery Reactive Power (kVAR)"] / df["Inverter Capacity (kVA)"]
    charging = df["Battery Electric Power (kW)"] > 0
    for i, (mask, label) in enumerate(((charging, "Charging"), (~charging, "Discharging"))):
        ax.scatter(
            df.loc[mask, "Voltage (-)"], q[mask],
            s=26, color=SERIES[i], edgecolor=SURFACE, linewidth=0.8, zorder=3,
            label=f"Battery, {label.lower()} (1-min steps)",
        )
    ax.axhline(0, color=MUTED, lw=0.8)
    for voltage, label, color in (
        (voltages[1], "Injection below", SERIES[0]),
        (voltages[2], "Absorption above", SERIES[1]),
    ):
        ax.axvline(voltage, color=color, ls=":", lw=1)
        ax.text(voltage, 0.97, f"{label} {voltage:g} p.u.",
                transform=ax.get_xaxis_transform(), rotation=90,
                ha="right", va="top", color=color, fontsize=7)
    ax.text(1.0, 0.3, "deadband", ha="center", color=TEXT_2, fontsize=8)
    ax.text(0.885, -0.3, "injects Q", color=TEXT_2, fontsize=8)
    ax.text(1.115, 0.3, "absorbs Q", ha="right", color=TEXT_2, fontsize=8)
    ax.set_xlim(0.88, 1.12)
    ax.set_ylim(-0.55, 0.55)
    ax.set_xlabel("House voltage received from GridLAB-D (p.u.)")
    ax.set_ylabel("Reactive power (p.u. of inverter kVA)")
    ax.set_title("Battery Reactive Power volt-VAr curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_timeseries(df):
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
    t = df.index
    voltages, _ = IEEE1547_CAT_B_VOLT_VAR

    ax = axes[0]
    ax.axhspan(voltages[1], voltages[2], color=BAND, zorder=0)
    for v in (voltages[0], voltages[-1]):
        ax.axhline(v, color=MUTED, ls=":", lw=1)
    ax.step(t, df["Swing Voltage (-)"], where="post", color=MUTED, ls="--", lw=1.2,
            label="Swing bus (player)")
    ax.step(t, df["Voltage (-)"], where="post", color=SERIES[0],
            label="House meter (GridLAB-D → OCHRE)")
    ax.text(t[-1], voltages[-1], "Q limit ", color=TEXT_2, fontsize=8, ha="right", va="bottom")
    ax.text(t[-1], voltages[0], "Q limit ", color=TEXT_2, fontsize=8, ha="right", va="top")
    ax.text(t[-1], 1.0, "deadband ", color=TEXT_2, fontsize=8, ha="right", va="center")
    ax.set_ylabel("p.u.")
    ax.set_title("Voltage")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2)

    ax = axes[1]
    expected = (
        np.interp(df["Voltage (-)"], *IEEE1547_CAT_B_VOLT_VAR) * df["Inverter Capacity (kVA)"]
    )
    ax.step(t, expected, where="post", color=MUTED, lw=4, alpha=0.5,
            label="Curve at received voltage")
    ax.step(t, df["Battery Reactive Power (kVAR)"], where="post", color=SERIES[0],
            label="Battery")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("kVAR")
    ax.set_title("Battery reactive power (+ absorb, − inject)")
    ax.legend(loc="upper right", ncol=2)

    ax = axes[2]
    ax.step(t, df["Battery P Setpoint (kW)"], where="post", color=MUTED, ls="--", lw=1.2,
            label="Setpoint")
    ax.step(t, df["Battery Electric Power (kW)"], where="post", color=SERIES[0],
            label="Delivered")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylim(-3, 3)
    ax.set_ylabel("kW")
    ax.set_title("Battery real power (+ charge, − discharge)")
    ax.legend(loc="upper left", ncol=2)
    format_time_axis(ax)
    mark_reactive_transitions(axes, df)

    fig.suptitle(
        f"OCHRE + GridLAB-D co-simulation: {df['Inverter Capacity (kVA)'].iloc[0]:g} kVA "
        "battery inverter on volt-VAr",
        x=0.02, ha="left", fontweight="bold", color=TEXT,
    )
    fig.tight_layout()
    return fig


def plot_exchange(df, half_window=dt.timedelta(minutes=6)):
    # zoom in on the largest step in the house Q published by OCHRE
    t_step = df["Total Reactive Power (kVAR)"].diff().abs().idxmax()
    zoom = df.loc[t_step - half_window : t_step + half_window]
    t = zoom.index

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    ax = axes[0]
    ax.step(t, zoom["Voltage (-)"], where="post", color=SERIES[0], marker="o", ms=4,
            label="Voltage received by OCHRE")
    ax.set_ylabel("p.u.")
    ax.set_title("House voltage (GridLAB-D → OCHRE, same step)")
    ax.legend(loc="best")

    for ax, (sent, received, unit, name) in zip(
        axes[1:],
        (
            ("Total Electric Power (kW)", "GLD Load P (kW)", "kW", "real power P"),
            ("Total Reactive Power (kVAR)", "GLD Load Q (kVAR)", "kVAR", "reactive power Q"),
        ),
    ):
        ax.step(t, zoom[sent], where="post", color=SERIES[0], marker="o", ms=4,
                label="Published by OCHRE")
        ax.step(t, zoom[received], where="post", color=SERIES[1], marker="s", ms=4,
                ls="--", label="Load solved by GridLAB-D")
        ax.set_ylabel(unit)
        ax.set_title(f"House {name} (OCHRE → GridLAB-D, one step later)")
        ax.legend(loc="best")
    format_time_axis(axes[-1], minutes=2)
    mark_reactive_transitions(axes, df, start=t[0], end=t[-1])

    fig.suptitle(
        f"HELICS data exchange around {t_step:%H:%M} (60 s steps)",
        x=0.02, ha="left", fontweight="bold", color=TEXT,
    )
    fig.tight_layout()
    return fig


def make_figures(output_dir):
    set_style()
    os.makedirs(output_dir, exist_ok=True)
    df = load_results()
    paths = []
    for name, plot in (
        ("cosim_volt_var_curve", plot_volt_var_curve),
        ("cosim_timeseries", plot_timeseries),
        ("cosim_exchange", plot_exchange),
    ):
        fig = plot(df)
        path = os.path.join(output_dir, f"{name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "figures")
    for p in make_figures(out):
        print("Saved", p)
