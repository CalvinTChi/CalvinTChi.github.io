---
layout: post
title:  "Collaborative Filtering"
date:   2020-09-24
categories: Classical_Machine_Learning
---

## 1. Introduction

Given historical ratings $R_{ij} \in \mathbb{R}$ from a set of $m$ users to a set of $p$ items, collaborative filtering is a framework for inferring an unknown rating for a given user, item pair from the historical ratings. There are multiple types of collaborative filtering approaches, and these are covered by this excellent [post](https://towardsdatascience.com/intro-to-recommender-system-collaborative-filtering-64a238194a26).

The topic of this post is on the matrix factorization approach, which finds the factorization $R = UV^{\top}$ for the usually sparse ratings matrix $R \in \mathbb{R}^{m \times p}$, where $U \in \mathbb{R}^{m \times d}$ and $V \in \mathbb{R}^{p \times d}$ with $d << m$ and $d << p$. Assuming an user's interests and an item's popularity exist in a lower-dimensional manifold, the $i$-th row $u_{i} \in \mathbb{R}^{d}$ from $U$ represents interests for user $i$ and the $j$-th row $v_{j} \in \mathbb{R}^{d}$ from $V$ represents popularity for item $j$. From $U, V$, the predicted rating of user $i$ for item $j$ is the dot product $\hat{r}_{ij} = u_{i}^{\top}v_{j}$. 

## 2. Method

For our derivation, we assume realistically that $R$ is sparse, where missingness can be denoted by the matrix $W \in \mathbb{R}^{m \times p}$ where

$$
\begin{align*}
W_{ij}=\begin{cases}
    1, & \text{if $R_{ij} \neq \texttt{NaN}$}\\
    0, & \text{Otherwise}\\
\end{cases}
\end{align*}
$$

The latent factors $U, V$ are learned by minimizing the following loss function

$$
\begin{align*}
\mathcal{L}(U, V, R) = \sum_{(i, j) \in S}(u_{i}^{\top}v_{j} - R_{ij})^{2} + \lambda \sum_{i = 1}^{m}||u_{i}||_{2}^{2} + \lambda \sum_{j = 1}^{p}||v_{j}||_{2}^{2}, \quad \text{where }S = \{(i, j): R_{ij} \neq \texttt{NaN}\}.
\end{align*}
$$

Unfortunately, since the $\mathcal{L}(U, V, R)$ is not jointly convex in $U, V$, we will need to employ alternating least squares to solve for $U$ and $V$. Alternating least squares solves for $U$ and $V$ in separate optimization steps holding the other set of latent factors constant. In this implementation, we will optimize the latent factor for one user and item at a time. Let $R_{u_{i}} \in \mathbb{R}^{p}$ be the item ratings from user $i$, $R_{v_{j}} \in \mathbb{R}^{m}$ be the ratings given to item $j$, $W_{u_{i}} = diag(W[i, :]) \in \mathbb{R}^{p \times p}$ be the diagonalized matrix of rating availability for user $i$, and $W_{v_{j}} = diag(W[:, j]) \in \mathbb{R}^{m \times m}$ is similarly defined as $W_{u_{i}}$ but for item $j$ instead. In matrix notation, the loss $\mathcal{L}$ for $u_{i}$ and $v_{j}$ are

$$
\mathcal{L}(R_{u_{i}}, u_{i}) = (R_{u_{i}} - Vu_{i})^{\top}W_{u_{i}}(R_{u_{i}} - Vu_{i}) + \lambda u_{i}^{\top}u_{i}
$$

$$
\mathcal{L}(R_{v_{j}}, v_{j}) = (R_{v_{j}} - Uv_{j})^{\top}W_{v_{j}}(R_{v_{j}} - Uv_{j}) + \lambda v_{j}^{\top}v_{j}.
$$

The closed form solution for $u_{i}$ is 

$$
\begin{align*}
\frac{\partial \mathcal{L}(R_{u_{i}}, u_{i})}{\partial u_{i}} &= \frac{\partial}{\partial u_{i}}\left(R_{u_{i}}^{\top}W_{u_{i}}R_{u_{i}} - R_{u_{i}}^{\top}W_{u_{i}}Vu_{i} - u_{i}^{\top}V^{\top}W_{u_{i}}R_{u_{i}} + u_{i}^{\top}V^{\top}W_{u_{i}}Vu_{i} + \lambda u_{i}^{\top}u_{i}\right)\\
&= -2V^{\top}W_{u_{i}}R_{u_{i}} + 2V^{\top}W_{u_{i}}Vu_{i} + 2\lambda u_{i} = 0\\
&\Rightarrow u_{i}^{*} = (V^{\top}W_{u_{i}}V + \lambda I_{d})^{-1}V^{\top}W_{u_{i}}R_{u_{i}}.
\end{align*}
$$

By a similar derivation, the closed form solution for $v_{j}$ is

$$
\begin{align*}
v_{j}^{*} = (U^{\top}W_{v_{j}}U + \lambda I_{d})^{-1}U^{\top}W_{v_{j}}R_{v_{j}}.
\end{align*}
$$


## 3. Implementation

We will apply collaborative filtering to build a joke recommender system based on historical joke ratings of $m = 24,983$ users for $p = 100$ jokes. Here, the ratings $R_{ij} \in [-10, 10]$, with a higher value representing higher satisfaction. Load libraries and data.


```python
from scipy import io
import numpy as np
import pandas as pd
DIR = "datasets"

train = pd.DataFrame(io.loadmat(DIR + '/joke_train.mat')['train'])
validation = open(DIR + '/validation.txt').readlines()
```

Preprocess the data. For training data, fill `NaN` with 0, and use 0 used to detect missingness in the implementation.


```python
train = train.to_numpy()
train[np.isnan(train)] = 0
validation = np.array([list(map(int, line.strip().split(','))) for line in validation])
```

Set the regularization parameter $\lambda = 125$ and number of latent features to learn $d = 10$


```python
lda = 125
d = 10
```

A few implementation details:

+ We will continue alternating least squares until the mean squared error portion of $\mathcal{L}(U, V, R)$ converges. 
+ Initialize each entry of $U, V$ with values drawn from $\mathcal{N}(0, 1)$. 


```python
def collaborative_filtering(R, d, lda, threshold):
    W = (R != 0).astype(np.float64)
    L1 = float("inf")
    # initialize user and joke latent factors
    V = np.random.normal(0, 1, (R.shape[1], d))
    U = np.random.normal(0, 1, (R.shape[0], d))
    # S_idx is tuple - 1st array contain row indices and 2nd array contains column indices
    S_idx = np.where(R != 0)
    Rhat = U.dot(V.T)
    L2 = np.mean((Rhat[S_idx] - R[S_idx])**2)
    iteration = 1
    while abs(L1 - L2) > threshold:
        L1 = L2
        # optimize U
        for i, Wu in enumerate(W):
            rating = R[i, :]
            U[i, :] = np.linalg.inv(V.T.dot(np.diag(Wu)).dot(V) + lda * np.eye(d)).dot(V.T.dot(np.diag(Wu)).dot(rating)).flatten()
        # optimize V
        for j, Wv in enumerate(W.T):
            rating = R[:, j]
            V[j, :] = np.linalg.inv(U.T.dot(np.diag(Wv)).dot(U) + lda * np.eye(d)).dot(U.T.dot(np.diag(Wv)).dot(rating)).flatten()
        Rhat = U.dot(V.T)
        L2 = np.mean((Rhat[S_idx] - R[S_idx])**2)
        print("Iteration {0}: Mean squared error is {1}".format(iteration, L2))
        iteration += 1
    return U, V
```

Train on historical data.


```python
U, V = collaborative_filtering(train, d, lda, 0.1)
```

    Iteration 1: Mean squared error is 18.325985391164423
    Iteration 2: Mean squared error is 12.870451586077216
    Iteration 3: Mean squared error is 12.091937379418232
    Iteration 4: Mean squared error is 11.697168710065604
    Iteration 5: Mean squared error is 11.4829143727792
    Iteration 6: Mean squared error is 11.357592793000398
    Iteration 7: Mean squared error is 11.27484206241288


The predicted ratings from learned latent factor matrices are $\hat{R} = UV^{\top}$. We can now use $u_{i}$ and $v_{j}$ to predict whether user $i$ will like joke $j$, with liking a joke given by $Y = \mathbb{1}(R_{ij} > 0)$. 

The validation dataset has format

```
1, 5, 1
1, 8, 1
```

Where each line is of the format `i, j, y`, where $i$ is user index, $j$ is joke index, and $y$ is whether user $i$ liked joke $j$. The indices from the validation dataset start with 1.


```python
Rhat = U.dot(V.T)
```

Convert ratings to whether to recommend joke to user. 


```python
Yhat = (Rhat > 0).astype(int)
```

Make prediction and calculate mean accuracy of recommendation.


```python
row_idx = validation[:, 0] - 1
col_idx = validation[:, 1] - 1

accuracy = np.mean(Yhat[row_idx, col_idx] == validation[:, 2])
print("Accuracy is {0}".format(round(accuracy, 3)))
```

    Accuracy is 0.72


This concludes the note.
