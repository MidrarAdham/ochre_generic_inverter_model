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