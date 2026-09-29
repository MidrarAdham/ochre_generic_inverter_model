# Project Title

Create a generic inverter model that can interacts with other OCHRE objects. OCHRE currently has an inverter object that only interact with the PV object.

The src folder contains other testing scripts. "test_dwelling.py" is an example of how the inverter model can be used with other OCHRE objects.

## Contributors
Midrar Adham

## Short Description

* **volt-VAR Curve** - provide IEEE 1547 volt-VAR curve to control the inverter.

* **volt-VAR Variation** - The curve can be updated during each time step of the simulation

* **Grid-voltage** - Already existed in OCHRE. It is a voltage value (PCC) that can be sent from an external tool to set the voltage at the inverter point.

## Tech Stack

* **Language:** Python

## Repository Contents

- src folder: contain all the files, including the inverter model code.

## Getting Started

Follow these steps to set up the project locally.

### Prerequisites

List any software, tools, or global packages needed:
* OCHRE
* Numpy
* python 3.12
* pytest, for model validation purposes.
* Poetry, not needed. But I use it for org

### Installation

1. **Clone the repository:**
   ```bash
   git clone git@github.com:PortlandStatePowerLab/midrar_ochre_generic_inverter_model_2026.git
   ```

### If I want to work on this project, where should I start from?
- Clone the repository
- Put the inverter.py file in the same location as your OCHRE scripts.
- In every OCHRE script you run, make sure you import the inverter file at the top of your script:
```
import inverter
```
- To run the tests, run:
```
python demo.py figures
```
### HELICS co-simulation test
The `cosim` folder connects an OCHRE dwelling (with an `InverterBattery` on the IEEE 1547 Cat. B volt-VAr curve) to a minimal GridLAB-D feeder through a HELICS broker. GridLAB-D sends the house voltage, and OCHRE sends back the house P + jQ. The swing voltage sweeps 0.90 -> 1.10 -> 0.90 p.u. over 4 hours at 1-minute steps.

Requires `gridlabd` (with the HELICS connection module) and `helics_broker` on PATH.
```
python cosim/run_cosim.py      # run once, results and logs in cosim/output/
pytest cosim/test_cosim.py     # run and check the results
python cosim/plot_cosim.py     # figures in cosim/figures/ (runs the co-simulation if needed)
```
