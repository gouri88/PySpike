# PySpike

An interactive implementation of the **Leaky Integrate-and-Fire (LIF) neuron** built from scratch in Python.

This project demonstrates the fundamental computational unit used in Spiking Neural Networks (SNNs). Unlike traditional Artificial Neural Networks, neurons communicate using discrete spikes over time, making SNNs more biologically realistic and energy efficient.

## Features

- LIF neuron implemented from scratch
- Interactive parameter tuning with Streamlit sliders
- Membrane potential visualization
- Red spike markers on threshold crossings
- Spike train visualization
- Firing rate analysis
- Frequency-Current (F-I) curve
- Time to first spike statistic
- Dark educational interface
- Animation mode that shows voltage evolving step by step
- No SNN frameworks used

## Technologies

- Python
- NumPy
- Matplotlib
- Streamlit

## Project Structure

```
PySpike/
|-- app.py              # Streamlit application
|-- neuron.py           # LIF neuron implementation
|-- simulation.py       # Simulation engine
|-- simulator.py        # Compatibility exports
|-- plots.py            # Plotting functions
|-- utils.py            # Statistics exports
|-- experiment.py       # Firing-rate experiment helper
|-- network.py          # Simple chained neuron network experiment
|-- main.py             # Command-line demo
|-- requirements.txt
|-- README.md
```

## Installation

```
pip install -r requirements.txt
```

## Run the App

```
streamlit run app.py
```

## Run the Command-Line Demo

```
python main.py
```
