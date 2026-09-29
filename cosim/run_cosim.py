"""
Run the OCHRE + GridLAB-D co-simulation: HELICS broker, GridLAB-D federate
(house.glm), and OCHRE federate (ochre_federate.py), each as its own process.

Logs and results go to cosim/output/. Usage:
    python cosim/run_cosim.py
"""

import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "output")
TIMEOUT = 600  # seconds

# GridLAB-D recorder complex values, rectangular (+1+2j) or polar in degrees (+240-0.2d)
GLD_COMPLEX = re.compile(r"^([+-]?[\d.]+(?:e[+-]?\d+)?)([+-][\d.]+(?:e[+-]?\d+)?)([ijd])$")


def parse_gld_complex(s):
    a, b, kind = GLD_COMPLEX.match(s.strip()).groups()
    a, b = float(a), float(b)
    if kind == "d":
        return a * np.exp(1j * np.radians(b))
    return complex(a, b)


def read_gld_recorder(file_name):
    df = pd.read_csv(file_name, comment="#", header=None, names=["Time", "Value"])
    t = pd.to_datetime(df["Time"].str.slice(0, 19))
    seconds = (t - t.iloc[0]).dt.total_seconds().astype(int)
    return pd.Series(df["Value"].map(parse_gld_complex).values, index=seconds)


def run(timeout=TIMEOUT):
    """Run the co-simulation and return {process name: return code}."""
    for exe in ("helics_broker", "gridlabd"):
        if shutil.which(exe) is None:
            raise RuntimeError(f"{exe} not found on PATH")
    shutil.rmtree(OUTPUT, ignore_errors=True)
    os.makedirs(OUTPUT)

    commands = {
        "broker": ["helics_broker", "-f", "2", "--coretype=zmq", "--loglevel=warning"],
        "gridlabd": ["gridlabd", "house.glm"],
        "ochre": [sys.executable, "-u", "ochre_federate.py"],
    }
    procs, logs = {}, []
    try:
        for name, cmd in commands.items():
            log = open(os.path.join(OUTPUT, f"{name}.log"), "w")
            logs.append(log)
            procs[name] = subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)

        deadline = time.time() + timeout
        for name, proc in procs.items():
            try:
                proc.wait(timeout=max(deadline - time.time(), 1))
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Co-simulation timed out waiting for {name}, see {OUTPUT}")
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        for log in logs:
            log.close()

    return {name: proc.returncode for name, proc in procs.items()}


if __name__ == "__main__":
    t0 = time.time()
    codes = run()
    print(f"Finished in {time.time() - t0:.0f} s, return codes: {codes}")
    print(f"Results and logs in {OUTPUT}")
    sys.exit(max(abs(c) for c in codes.values()))
