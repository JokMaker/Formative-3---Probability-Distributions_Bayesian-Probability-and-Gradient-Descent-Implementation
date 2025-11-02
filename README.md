# Machine Learning Formative Assignment 3

A comprehensive exploration of statistical distributions, Bayesian probability, and gradient descent optimization implemented from scratch using real-world datasets.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Part 1: Bivariate Normal Distribution](#part-1-bivariate-normal-distribution)
- [Part 2: Bayesian Probability](#part-2-bayesian-probability)
- [Part 3: Gradient Descent Manual Calculation](#part-3-gradient-descent-manual-calculation)
- [Part 4: Gradient Descent Implementation](#part-4-gradient-descent-implementation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Key Results](#key-results)
- [Visualizations](#visualizations)

## Overview

This project demonstrates fundamental machine learning and statistical concepts through hands-on implementation:

1. **Bivariate Normal Distribution**: Implemented from scratch using NumPy only (no statistical libraries)
2. **Bayesian Probability**: Applied Bayes' theorem to sentiment analysis
3. **Gradient Descent**: Both manual calculations and Python implementations (manual and SciPy)

## Project Structure

```
Formative3/
│
├── Formative3_ML.ipynb          # Main Jupyter notebook with all implementations
├── MPI_national.csv              # Multidimensional Poverty Index dataset
├── IMDB Dataset.csv              # Movie review sentiment dataset
└── README.md                     # This file
```

## Features

- **From-scratch implementations** using only NumPy for core mathematical operations
- **Real-world datasets**: African countries MPI data and IMDB movie reviews
- **Comprehensive visualizations**: Contour plots, 3D surfaces, and probability comparisons
- **Step-by-step gradient descent** with detailed iteration tracking
- **Multiple optimization approaches**: Manual gradient descent and SciPy optimization

## Part 1: Bivariate Normal Distribution

### Objective
Implement bivariate normal distribution from scratch and visualize with contour and 3D plots using African countries MPI data.

### Dataset
- **Source**: Multidimensional Poverty Index (MPI) data
- **Variables**: 
  - X: MPI Urban (Urban poverty index)
  - Y: MPI Rural (Rural poverty index)
- **Sample Size**: 46 African countries

### Implementation Details

**Mathematical Formula:**
```
f(x,y) = 1/(2π·σx·σy·√(1-ρ²)) · exp(-Q/2)

where Q = 1/(1-ρ²) · [(x-μx)²/σx² - 2ρ(x-μx)(y-μy)/(σx·σy) + (y-μy)²/σy²]
```

**Key Results:**
- Urban mean (μₓ): 0.143804
- Rural mean (μᵧ): 0.368522
- Correlation (ρ): 0.873629
- Strong positive correlation between urban and rural poverty indices

**Visualizations:**
- Contour plots showing probability density
- 3D surface plots of the distribution
- Data points overlaid on distribution

## Part 2: Bayesian Probability

### Objective
Apply Bayesian probability to analyze movie review sentiment using keywords from the IMDB dataset.

### Dataset
- **Source**: IMDB Movie Reviews Dataset
- **Size**: 50,000 reviews
- **Classes**: Positive and Negative sentiment

### Selected Keywords
- `love`
- `violence`
- `wonderful`
- `likable`

### Bayesian Analysis

**Probabilities Calculated:**
- Prior: P(Positive) = 0.5000
- Likelihood: P(Keyword | Positive)
- Marginal: P(Keyword)
- **Posterior: P(Positive | Keyword)** ← Main focus

**Key Findings:**
- Positive keywords (`love`, `wonderful`, `likable`) show high posterior probabilities (>0.5)
- Negative keyword (`violence`) shows lower posterior probability
- Bayes' theorem successfully updates prior beliefs with keyword evidence

### Visualizations
- Bar charts comparing posterior probabilities
- Side-by-side probability comparisons (likelihood, marginal, posterior)

## Part 3: Gradient Descent Manual Calculation

Manual step-by-step calculations of gradient descent for linear regression:
- Iteration 1: [Page 1](https://drive.google.com/file/d/1YI1N-citx1IfYYEj9A73BRb0bbnmyDET/view?usp=sharing) & [Page 2](https://drive.google.com/file/d/153STO_oG0QivjDEk84lRV0x5AfTUvXZ6/view?usp=sharing)
- Iteration 2: [Link](https://drive.google.com/file/d/1v53tPSbX4jpwJgJLchBkW5K4Hjnnw5YY/view?usp=sharing)
- Iteration 3: [Link](https://drive.google.com/file/d/1lZAeJxxVtYJ-QRHBjcndGQHrwNgyGXRJ/view?usp=sharing)
- Iteration 4: [Page 1](https://drive.google.com/file/d/1M5MZPirebJ-EjLZpdO0KdSKKMIjXIJCh/view?usp=drive_link) & [Page 2](https://drive.google.com/file/d/1v05Z7_cqb44PyIimDgd-xa6rA0KaZ21-/view?usp=drive_link)

## Part 4: Gradient Descent Implementation

### Problem Specification
- **Model**: Linear regression `y = mx + b`
- **Initial Parameters**: m₀ = -1, b₀ = 1
- **Learning Rate**: α = 0.1
- **Data Points**: (1,3) and (3,6)
- **Cost Function**: J(m,b) = (1/n)Σ(yᵢ - (mxᵢ + b))²

### Implementation Approaches

#### 4A: Manual Gradient Descent
- Step-by-step gradient computation
- Parameter updates with explicit formulas
- Full iteration tracking

**Results:**
- Initial MSE: 36.500000
- Final MSE: 0.034816
- **99.90% error reduction**
- Final parameters: m = 1.333600, b = 1.896800

#### 4B: SciPy Optimization
- Uses `scipy.optimize.minimize` with BFGS method
- Provides optimal solution
- Final MSE: 0.000000 (near perfect fit)
- Optimal parameters: m = 1.500000, b = 1.500000

### Visualizations
- Parameter evolution (m and b) over iterations
- Error (MSE) reduction over iterations
- Detailed iteration summary tables

## Requirements

### Python Libraries
```python
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scipy>=1.7.0
jupyter>=1.0.0
```

### Installation
```bash
pip install numpy pandas matplotlib scipy jupyter
```

### Datasets
Ensure the following CSV files are in the same directory as the notebook:
- `MPI_national.csv` - Multidimensional Poverty Index data
- `IMDB Dataset.csv` - Movie review sentiment data

## Usage

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure datasets are available** (MPI_national.csv and IMDB Dataset.csv)
4. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook Formative3_ML.ipynb
   ```
5. **Run cells sequentially** to reproduce all results

## Key Results Summary

### Part 1: Bivariate Normal Distribution
- Successfully implemented PDF from scratch
- Strong correlation (ρ = 0.874) between urban and rural poverty
- Comprehensive visualizations created

### Part 2: Bayesian Probability
- Demonstrated Bayesian inference on text sentiment
- Keywords like "love" and "wonderful" strongly predict positive sentiment
- Keywords like "violence" are associated with negative sentiment

### Part 3 & 4: Gradient Descent
- Manual implementation achieved 99.90% error reduction
- SciPy optimization achieved near-perfect fit (MSE ≈ 0)
- Clear demonstration of convergence behavior

## Visualizations

The notebook includes multiple high-quality visualizations:

1. **Bivariate Normal Distribution**:
   - Contour plots with filled regions
   - 3D surface plots
   - Combined multi-panel views

2. **Bayesian Analysis**:
   - Posterior probability bar charts
   - Comparative probability visualizations
   - Color-coded sentiment indicators

3. **Gradient Descent**:
   - Parameter convergence plots
   - Error reduction curves
   - Iteration-by-iteration progress tracking

## Methodology Highlights

### From-Scratch Implementation
- No statistical libraries used for core calculations
- Only NumPy for basic array operations
- Direct implementation of mathematical formulas

### Educational Focus
- Step-by-step explanations
- Clear mathematical formulations
- Detailed iteration tracking
- Multiple implementation approaches for comparison

## Notes

- All implementations prioritize **clarity** and **educational value**
- Manual calculations provide deep understanding of underlying mathematics
- SciPy comparison shows optimization library capabilities
- Real-world datasets ensure practical relevance

## Learning Outcomes

This project demonstrates:
- Understanding of multivariate probability distributions
- Bayesian probability and inference
- Gradient descent optimization
- Linear regression fundamentals
- Data visualization skills
- Real-world data analysis

