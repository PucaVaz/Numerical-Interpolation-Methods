# Numerical Analysis Project: Data Fitting and Interpolation

A comprehensive implementation of numerical methods for data fitting and interpolation, providing both mathematical implementations and an interactive visualization dashboard.

![Numerical Analysis Methods](https://github.com/user-attachments/assets/placeholder-image-id)

## Overview

**Numerical Analysis Project** implements several mathematical approaches for analyzing relationships between variables, with a focus on ice cream sales data. The project demonstrates both regression techniques (for approximating trends) and interpolation methods (for estimating values between known data points).

This project is part of the course "Numerical Analysis" taught by Prof. Dr. Moisés Dantas. More details about the professor can be found on his [LinkedIn profile](https://www.linkedin.com/in/moises-santos-2b72a37a/).

## Features

- **Least Squares Regression (MMQ)**
  - Linear regression model
  - Quadratic regression model
  
- **Interpolation Techniques**
  - Newton's Divided Differences method
  - Lagrange Polynomial method
  
- **Interactive Dashboard**
  - Method selection interface
  - Parameter adjustment controls
  - Real-time visualization of results
  
- **Comprehensive Visualization**
  - Plots comparing original data with mathematical models
  - Clear visualization of different numerical approaches

## Installation

### Prerequisites
- Python 3.11+
- Conda package manager

### Setup
1. Clone the repository to your local machine
2. Create and activate the Conda environment:
```
conda env create -f environment.yml
conda activate numerical-interpolation-methods
```

## Usage

### Running Individual Mathematical Implementations

To explore specific mathematical implementations:

**Least Squares Method (MMQ):**
```
python -m src.mathImplementations.mmq
```

**Lagrange Interpolation:**
```
python -m src.mathImplementations.lagrangeInterpol
```

**Newton's Divided Differences Interpolation:**
```
python -m src.mathImplementations.newtonInterpol
```

### Interactive Dashboard

To launch the interactive Gradio dashboard that allows for method selection and parameter adjustment:
```
python -m src.dasboard.gradio_app
```

## Project Structure

- **src/mathImplementations/**
  - `mmq.py`: Implementation of Least Squares Method for regression
  - `newtonInterpol.py`: Implementation of Newton's Divided Differences interpolation
  - `lagrangeInterpol.py`: Implementation of Lagrange Polynomial interpolation
  
- **src/dashboard/**
  - `gradio_app.py`: Interactive dashboard for visualizing methods
  
- **src/utils/**
  - `helpers.py`: Utility functions for data loading and manipulation
  
- **data/**
  - `ice_cream.csv`: Dataset containing temperature and revenue data

## Mathematical Background

### Least Squares Regression
The project implements both linear (y = b + a·x) and quadratic (y = b + a·x²) regression models using the Least Squares Method. This approach minimizes the sum of squared differences between observed and predicted values.

### Interpolation Methods
Two classic interpolation techniques are implemented:

- **Newton's Divided Differences**: Builds a polynomial using a recursive formula based on divided differences
- **Lagrange Interpolation**: Creates a polynomial where each term is constructed to be zero at all data points except one

Both methods produce polynomials that pass exactly through all given data points, but their mathematical formulations and computational approaches differ significantly.

## Implementation Details

### Symbolic Computation
The project uses SymPy to create symbolic mathematical expressions, which are then converted to numerical functions for evaluation and plotting.

### Visualization
Matplotlib is used to generate clear comparisons between original data points and the mathematical models, helping to visualize the effectiveness of different approaches.

### Interactive Interface
The Gradio dashboard provides an intuitive way to:
- Select between different mathematical methods
- Adjust parameters (such as the number of nodes for interpolation)
- Visualize results in real-time

## Troubleshooting

If you encounter issues:
- Verify that the dataset (`data/ice_cream.csv`) is present in the correct location
- Ensure all dependencies are properly installed via the conda environment
- Check console output for specific error messages

## Acknowledgments

This project was developed as part of the Numerical Analysis course under the guidance of Prof. Dr. Moisés Dantas.