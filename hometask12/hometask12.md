```python
import numpy as np
import pandas as pd
from scipy.io import loadmat
```


```python
def get_data(file_path='ex8_movies.mat'):
    mat = loadmat(file_path)

    Y = pd.DataFrame(mat['Y']).astype(float).values
    R = pd.DataFrame(mat['R']).astype(float).values

    return Y, R

Y, R = get_data()
print('Y.shape =', Y.shape)
print('R.shape =', R.shape)
```

    Y.shape = (1682, 943)
    R.shape = (1682, 943)
    


```python
def J(Y, R, X, Theta, lambd):
    assert X.shape[1] == Theta.shape[0]

    h = X @ Theta

    error_cost = 1/2 * np.sum((h - Y) * R) ** 2
    reg_X = lambd / 2 * np.sum(X ** 2)
    reg_Theta = lambd / 2 * np.sum(Theta ** 2)

    return error_cost + reg_X + reg_Theta
```


```python
def J_derivative(Y, R, X, Theta, lambd):
    cost_matr = (X @ Theta - Y) * R

    X_grad = cost_matr @ Theta.T + lambd * X
    Theta_grad = (cost_matr.T @ X).T + lambd * Theta

    return X_grad, Theta_grad
```


```python
def fit(Y, R, num_features=10, alpha=0.001, lambd=1.0, eps=0.1, max_iter=2000, step=100, verbose=1):
    num_movies, num_users = Y.shape

    if verbose:
        print(f'Ejecutando descenso de gradiente con alpha={alpha}, lambda={lambd}, eps={eps}, max_iter={max_iter}')

    np.random.seed(2019)
    X = np.random.randn(num_movies, num_features)
    Theta = np.random.randn(num_features, num_users)

    J_hist = [-1]
    iter_number = 0

    while iter_number <= max_iter:
        X_grad, Theta_grad = J_derivative(Y, R, X, Theta, lambd)

        X -= alpha * X_grad        
        Theta -= alpha * Theta_grad

        current_J = J(Y, R, X, Theta, lambd)
        J_hist.append(current_J)

        if iter_number > 0 and np.abs(J_hist[iter_number-1] - J_hist[iter_number]) < eps: 
            print(f'Se han alcanzado los criterios de parada. J_hist[{iter_number}]={J_hist[iter_number]}')
            break

        iter_number += 1

        if verbose and iter_number % step == 0:
            print(f'Iteration {iter_number}: Cost = {J_hist[-1]}')

    return X, Theta, J_hist

X, Theta, J_hist = fit(Y, R, alpha=0.001, lambd=1, max_iter=2000, verbose=1)
```

    Ejecutando descenso de gradiente con alpha=0.001, lambda=1, eps=0.1, max_iter=2000
    Iteration 100: Cost = 2360238.0881440127
    Iteration 200: Cost = 141567754.58423796
    Iteration 300: Cost = 59333571.54532845
    Iteration 400: Cost = 46945290.99708298
    Iteration 500: Cost = 42377257.910620615
    Iteration 600: Cost = 40527985.590354435
    Iteration 700: Cost = 39555230.85736203
    Iteration 800: Cost = 38829093.548187874
    Iteration 900: Cost = 38137797.67578375
    Iteration 1000: Cost = 37480101.83008587
    Iteration 1100: Cost = 36896758.30429841
    Iteration 1200: Cost = 36398218.80242452
    Iteration 1300: Cost = 35969296.15288059
    Iteration 1400: Cost = 35590956.23935748
    Iteration 1500: Cost = 35251666.343423285
    Iteration 1600: Cost = 34947125.450240046
    Iteration 1700: Cost = 34675422.836912386
    Iteration 1800: Cost = 34433373.037404545
    Iteration 1900: Cost = 34215788.13215034
    Iteration 2000: Cost = 34016571.943543814
    


```python
pred = X @ Theta
my_pred = pred[:, 0]
top_pred = np.argsort(my_pred)[::-1]

print('Los 10 mejores índices de recomendaciones de películas para el usuario 0:')
for i in range(10):
    idx = top_pred[i]
    print(f'Prediciendo la calificación {my_pred[idx]:.2f} para la película con ID {idx}')
```

    Los 10 mejores índices de recomendaciones de películas para el usuario 0:
    Prediciendo la calificación 7.45 para la película con ID 49
    Prediciendo la calificación 5.39 para la película con ID 918
    Prediciendo la calificación 5.37 para la película con ID 407
    Prediciendo la calificación 5.36 para la película con ID 168
    Prediciendo la calificación 5.34 para la película con ID 511
    Prediciendo la calificación 5.28 para la película con ID 901
    Prediciendo la calificación 5.27 para la película con ID 855
    Prediciendo la calificación 5.27 para la película con ID 55
    Prediciendo la calificación 5.27 para la película con ID 646
    Prediciendo la calificación 5.26 para la película con ID 113
    
