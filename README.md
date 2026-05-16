# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
### Date: 

### AIM:
To implement ARMA model in python.

### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.

### PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
```

```
data = pd.read_csv('/content/education_vs_income_econometrics_samples.csv')
```

```
N=1000
plt.rcParams['figure.figsize'] = [12, 6]
X=data['x_education_years']
plt.plot(X)
plt.title('Original Data')
plt.show()
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data ACF')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()
```

```
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']
```

```
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()
```

```
plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()
```

```
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']
```

```
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()
```

```
plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()
```

### OUTPUT:

ORIGINAL DATA:

<img width="977" height="528" alt="image" src="https://github.com/user-attachments/assets/f8c64758-e391-4799-9a4e-4b167c7fb29d" />

ORIGINAL DATA ACF & ORIGINAL DATA PACF:

<img width="1198" height="590" alt="image" src="https://github.com/user-attachments/assets/28b2226e-9210-4cc1-a6ff-d06e15029456" />

SIMULATED ARMA(1,1) PROCESS:

<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/8677da95-e61b-4638-8e53-9002ea72bc22" />

Partial Autocorrelation

<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/122abef6-d52e-409d-b53e-a2b2d81806c0" />

Autocorrelation

<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/4c23ff1d-00fe-4833-a9f9-e42bb7c6b6b8" />

SIMULATED ARMA(2,2) PROCESS:

<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/76687e49-f55a-4003-a27a-4dc6660f738a" />

Partial Autocorrelation

<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/10431442-2185-456c-8708-04b5a53edfe6" />

Autocorrelation

<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/b879b2b8-a5a6-4599-ab33-7356897c023c" />

### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
