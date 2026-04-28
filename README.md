# Exploring Common Tests for Gaussianity of Data Sets

This repository contains the implementation and analysis of various statistical tests used to determine if a dataset follows a Gaussian (Normal) distribution. It features both manual Python implementations of the algorithms and comparisons with established scientific libraries.

## 📊 Overview
Testing for normality is a fundamental step in statistical analysis and data science. This project evaluates four classic tests using professional-grade astronomical datasets to observe how these tests perform in both ideal and non-ideal scenarios.


## 🛠️ Statistical Tests Implemented
The following tests are implemented manually in `main.ipynb` to demonstrate the underlying mathematics, then benchmarked against `scipy.stats`:

* **Kolmogorov-Smirnov (K-S) Test**: Calculates the maximum distance between the empirical distribution of the sample and the cumulative distribution function of the reference distribution.
* **Shapiro-Wilk (W) Test**: Evaluates the correlation between the data and the ideal normal scores.
* **Anderson-Darling (A^2) Test**: A more sensitive version of the K-S test that gives additional weight to the "tails" of the distribution.
* **Pearson Chi-Squared Test**: A goodness-of-fit test that compares observed binned frequencies against expected theoretical frequencies.



## 🔭 Datasets
To validate the tests, two real-world datasets are retrieved using `astropy` and `astroquery`:

1.  **Gaussian Case: Planck Satellite Data** A central slice of the Cosmic Microwave Background (CMB) map (Planck LFI). The temperature fluctuations of the CMB are famously Gaussian.
2.  **Non-Gaussian Case: SDSS Galaxy Redshifts** Redshift data from the Sloan Digital Sky Survey. Large-scale structure distributions typically exhibit non-Gaussian characteristics.

## 💻 Technical Stack
* **Language**: Python 3.13
* **Environment**: Jupyter Notebook
* **Key Libraries**: 
    * `numpy` & `pandas`: Data manipulation.
    * `scipy.stats`: Statistical benchmarking.
    * `matplotlib`: Distribution visualization.
    * `astropy` & `astroquery`: Astronomical data retrieval and FITS file handling.
    * `tabulate`: Clean console output for results.

## 🚀 How to Run
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/Jaidhar-689/Exploring-Gaussianity-Tests.git](https://github.com/Jaidhar-689/Exploring-Gaussianity-Tests.git)
