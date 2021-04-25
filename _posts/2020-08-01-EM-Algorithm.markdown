---
layout: post
title:  "Expectation-Maximization Algorithm"
date:   2020-08-01
categories: statistics
---

## 1. Motivation

The standard maximum likelihood estimation (MLE) problem solves for

$$\theta^{*} = \arg \max_{\theta}\ell(\theta),$$

the parameter that maximizes the log-likelihood of observed data $$\ell(\theta)$$, given a statistical model. However, in incomplete data scenarios with unobserved latent variable $$Z$$, simultaneously solving for $$z, \theta$$ to maximize the log-likelihood $$\ell(\theta, z)$$ can be impossible. Conversely, if $$z$$ were observed, the estimation problem would be easily solvable. 

The expectation-maximization (EM) algorithm is a method to solve for a local maximum likelihood estimate of $$\theta$$ numerically in incomplete data scenarios, alternating maximization between the two sets of unknowns, keeping the other set fixed. This idea is also known as coordinate ascent. 

## 2. Algorithm

The following presentation is largely based on the notes written by Andrew Ng [1]. Given a dataset of $$\{x^{(1)},...,x^{(m)}\}$$ of $$m$$ independent samples, the log-likelihood is given by

$$\ell(\theta) = \sum_{i = 1}^{m}\log p(x^{(i)}; \theta) = \sum_{i = 1}^{m}\log \sum_{z^{(i)}}p(x^{(i)}, z^{(i)}; \theta),$$

where $$Z$$ is an unknown discrete random variable ($$z$$'s are outcome values). Now for any distribution $$z^{(i)} \sim Q_{i}$$ (i.e. $$Q_{i}(z^{(i)})$$), we can further rewrite $$\ell(\theta)$$

$$
\begin{align*}
\ell(\theta) &= \sum_{i = 1}^{m}\log \sum_{z^{(i)}}p(x^{(i)}, z^{(i)}; \theta)\\
&= \sum_{i=1}^{m}\log\sum_{z^{(i)}}Q_{i}(z^{(i)})\frac{p(x^{(i)}, z^{(i)};\theta)}{Q_{i}(z^{(i)})}\\
&\geq \sum_{i=1}^{m}\sum_{z^{(i)}}Q_{i}(z^{(i)})\log\frac{p(x^{(i)}, z^{(i)};\theta)}{Q_{i}(z^{(i)})}.
\end{align*}
$$

The inequality above is due to Jensen's inequality applied to concave functions. Jensen's inequality states that for a convex function $$f$$ and random variable $$X$$, the following inequality $$\mathbb{E}[f(X)] \geq f(\mathbb{E}[X])$$ is true. One way to easily recall the direction of inequality is using the variance formula $$Var(X) = \mathbb{E}[X^{2}] - \mathbb{E}[X]^{2} \geq 0 \iff \mathbb{E}[X^{2}] \geq \mathbb{E}[X]^{2}$$, where $$f(x) = x^{2}$$ is a convex function. 

Now the expectation and maximization steps can be derived. The expectation step considers current $$\theta$$ value fixed and sets $$Q_{i}(z^{(i)})$$ so that the inequality above becomes equality. Start by noticing that

$$\log\sum_{z^{(i)}}\frac{p(z^{(i)}, x^{(i)};\theta)}{Q_{i}(z^{(i)})}Q_{i}(z^{(i)}) = g\Big(\mathbb{E}\Big[\frac{p(z^{(j)}, x^{(i)};\theta)}{Q_{i}(z^{(i)})}\Big]\Big)$$

and

$$\sum_{z^{(i)}}Q_{i}(z^{(i)})\log\frac{p(z^{(i)}, x^{(i)};\theta)}{Q_{i}(z^{(i)})} = 
\mathbb{E}\Big[g\left(\frac{p(z^{(i)}, x^{(i)};\theta)}{Q_{i}(z^{(i)})}\right)\Big],$$

which are summation terms to the left and right of the inequality respectively. Jensen's inequality for concave functions state $$g\Big(\mathbb{E}\Big[\frac{p(z^{(j)}, x^{(i)};\theta)}{Q_{i}(z^{(i)})}\Big]\Big) \geq \mathbb{E}\Big[g\left(\frac{p(z^{(i)}, x^{(i)};\theta)}{Q_{i}(z^{(i)})}\right)\Big]$$. It is easy to see that in order for equality to be achieved, $$\frac{p(z^{(j)}, x^{(i)};\theta)}{Q_{i}(z^{(i)})} = c$$ for some constant $$c$$, since $$g(\mathbb{E}[c]) = g(c) = \mathbb{E}[g(c)]$$. Choosing $$Q_{i}(z^{(i)}) \propto p(x^{(i)}, z^{(i)}; \theta)$$ is sufficient to achieve constant value. For $$Q_{i}$$ to remain a probability distribution, set

$$
\begin{align}
Q_{i}(z^{(i)}) &= \frac{p(x^{(i)}, z^{(i)};\theta)}{\sum_{z^{(i)}}p(x^{(i)}, z^{(i)};\theta)}\\
&=\frac{p(x^{(i)}, z^{(i)};\theta)}{p(x^{(i)};\theta)}\\
&=p(z^{(i)}|x^{(i)};\theta),
\end{align}
$$

which can be computed from $$p(x^{(i)} \lvert z^{(i)}; \theta)$$ using Bayes rule. In the maximization step, we hold $$Q_{i}(z^{(i)})$$ fixed and maximize the lower bound $$\sum_{i=1}^{m}\sum_{z^{(i)}}Q_{i}(z^{(i)})\log\frac{p(x^{(i)}, z^{(i)};\theta)}{Q_{i}(z^{(i)})}$$ with respect to $$\theta$$. The algorithm can succinctly be summarized below:

> Repeat until convergence {
>
> $$\hspace{1.5cm}$$Expectation step: for each $$i$$, set 
> $$Q_{i}(z^{(i)}) := p(z^{(i)}|x^{(i)};\theta)$$
>
> $$\hspace{1.5cm}$$Maximization step: set
> $$\theta := \arg\max_{\theta} \sum_{i=1}^{m}\sum_{z^{(i)}}Q_{i}(z^{(i)})log\frac{p(x^{(i)}, z^{(i)};\theta)}{Q_{i}(z^{(i)})}$$
>
>}

Note that the first step is called the expectation step because choosing $$Q_{i}$$ enables $$\mathbb{E}_{z^{(i)} \sim Q_{i}}\Big[g\left(\frac{p(z^{(i)}, x^{(i)};\theta)}{Q_{i}(z^{(i)})}\right)\Big]$$ to be defined. 

There is a perspective viewing the EM algorithm as coordinate ascent on the lowerbound of the log-likelihood by maximizing it with respect to $$Q$$ in the expectation step and then maximizing it with respect to $$\theta$$ in the maximization step. This becomes clear if we write $$\ell(\theta)$$ as

$$J(Q, \theta) = \sum_{i=1}^{m}\sum_{z^{(i)}}Q_{i}(z^{(i)})\log\frac{p(x^{(i)}, z^{(i)};\theta)}{Q_{i}(z^{(i)})}.$$

## 2. Proof of Convergence

To show that the EM algorithm convergences to some local optimum $$\theta^{*}$$ after $n$ iterations, we need to show that $$\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})$$, where $$t$$ represents the $$t$$-th iteration of EM algorithm. 

After one iteration of EM algorithm, we achieve the following inequality

$$
\begin{align*}
l(\theta^{(t+1)}) &\ge \sum_{i=1}^{m}\sum_{z^{(i)}}Q_{i}^{(t)}(z^{(i)})log\frac{p(z^{(i)}, x^{(i)};\theta^{(t+1)})}{Q_{i}^{(t)}(z^{(i)})}\\
&\geq \sum_{i=1}^{m}\sum_{z^{(i)}}Q_{i}^{(t)}(z^{(i)})\log\frac{p(z^{(i)}, x^{(i)};\theta^{(t)})}{Q_{i}^{(t)}(z^{(i)})} = \ell(\theta^{(t)}),
\end{align*}
$$

where the first inequality is due to Jensen's inequality, the second inequality is due to setting $$\theta^{(t + 1)} = \arg\max_{\theta}\sum_{i=1}^{m}\sum_{z^{(i)}}Q_{i}^{(t)}(z^{(i)})\log\frac{p(z^{(i)}, x^{(i)};\theta)}{Q_{i}^{(t)}(z^{(i)})}$$, and the third equality is due to choosing $$Q_{i}(z^{(i)}) = p(z^{(i)}\lvert \:x^{(i)};\theta)$$ in the expectation step.


## 3. Examples

### 3.1 Coin Bias Estimation

The following example is based on the one provided by Do and Batzoglou [2]. Suppose we wish to estimate the biases (probability of flipping heads) $$\theta_{A}$$ and $$\theta_{B}$$ of two coins $$A$$ and $$B$$ respectively, given independent repeated sets of 10 coin tosses, where the identity of the coin used ($$A$$ or $$B$$) used in each set of tosses is unknown.

<p align="center">
  <img src="https://i.imgur.com/6mvD1bt.png"/>
</p>

The first step is to define the statistical model. Let $$n$$ be number of coin tosses in a set and $$m$$ be number of sets. Also, let 

$$
\begin{align*}
x^{(i)} &= \text{number of heads in the } i\text{th set of 10 tosses}\\
z^{(i)} &= \text{whether coin used was } A \text{ or } B, \text{in $i$-th set of 10 tosses, where } z^{(i)} \in \{A, B\}\\
Q_{i}(z^{(i)}) &= \text{probability of coin } z^{(i)} \text{ given }i\text{-th set of 10 tosses}.\\
\theta &= \{\theta_{A}, \theta_{B}\}\\
\end{align*}
$$

The probability of coin $$z$$ is modeled as $$z \sim \text{bernoulli($\phi$)}$$ and $$x \lvert z \sim \text{binomial}(n, \theta_{z})$$, where $$p(z = A \lvert \phi) = \phi$$ and $$p(z = B \lvert \phi) = 1 - \phi$$.

### 3.1.1 Expectation
Recall that in the expectation step set

$$
\begin{align*}
Q_{i}(z^{(i)}) &= p(z^{(i)}|x^{(i)};\theta)\\
&= \frac{p(x^{(i)}, z^{(i)};\theta)}{\sum_{z^{(i)}}p(x^{(i)}, z^{(i)};\theta)}\\
&= \frac{p(x^{(i)} | z^{(i)};\theta)p(z^{(i)}; \theta)}{\sum_{z^{(i)}}p(x^{(i)} | z^{(i)};\theta)p(z^{(i)}; \theta)},
\end{align*}
$$

where the last line comes from applying Bayes rule. Thus, for coin $$A$$

$$Q_{i}(A) = \frac{\theta_{A}^{x^{(i)}}(1-\theta_{A})^{n-x^{(i)}}\phi}{\theta_{A}^{x^{(i)}}(1-\theta_{A})^{n-x^{(i)}}\phi + \theta_{B}^{x^{(i)}}(1-\theta_{B})^{n-x^{(i)}}(1 - \phi)}$$

The term $$Q_{i}(B)$$ is derived analogously.

### 3.1.2 Maximization
We maximize the log-likelihood lowerbound with respect to $$\theta_{A}, \theta_{B}$$, and $$\phi$$ using the first-order optimality condition. To find $$\theta_{A}$$, express $$\ell(\theta, \phi)$$ only in terms that contain $$\theta_{A}$$

$$
\begin{align*}
\ell(\theta, \phi) \propto \sum_{i=1}^{m}Q_{i}(A)\left(x^{(i)}\log\theta_{A} + (n - x^{(i)})\log(1 - \theta_{A}) - \log Q_{i}(A)\right).
\end{align*}
$$

Then, take partial derivative with respect to $$\theta_{A}$$ and solve for $$\theta_{A}$$ so that the partial derivative is equal to zero.

$$
\begin{align*}
\frac{\partial}{\partial\theta_{A}}\ell(\theta, \phi) &= \sum_{i=1}^{m}Q_{i}(z^{(i)}) \left(\frac{x^{(i)}}{\theta_{A}} - \frac{n - x^{(i)}}{1-\theta_{A}}\right) = 0\\
&\iff \theta_{A}^{*} = \frac{\sum_{i = 1}^{m}Q_{i}(z^{(i)})x^{(i)}}{n\sum_{i = 1}^{m}Q_{i}(z^{(i)})}.
\end{align*}
$$

The process to finding $$\theta_{B}^{*}$$ is analogous. Next, to find $$\phi$$, express $$\ell(\theta, \phi)$$ only in terms that contain $$\phi$$

$$
\begin{align*}
\ell(\theta, \phi) \propto \sum_{i=1}^{m}Q_{i}(A)\log \phi + Q_{i}(B)\log(1 - \phi)
\end{align*}
$$

Then, take partial derivative with respect to $$\phi$$ and solve for $$\phi$$ so that the partial derivative is equal to zero.

$$
\begin{align*}
\frac{\partial}{\partial\phi}\ell(\theta, \phi) &= \sum_{i=1}^{m}\frac{Q_{i}(A)}{\phi} - \frac{Q_{i}(B)}{1 - \phi} = 0\\
&\iff \phi^{*} = \frac{\sum_{i = 1}^{m}Q_{i}(A)}{\sum_{i = 1}^{m}Q_{i}(A) + Q_{i}(B)}
\end{align*}
$$

### 3.1.3 Implementation
Now we can implement the EM algorithm to estimate the biases of two coins given a set of coin toss results. We will first implement the EM algorithm class with the following usage:

```
EM_coin(initial=(0.400, 0.600, 0.500), n = 10, convergence=0.001, verbose=False)
```

where

1. `initial`: tuple of size three indicating initial guesses to the coin biases and coin probabilities.
2. `convergence`: threshold of sum of differences in estimated parameters between iterations, below which the EM algorithm stops.
3. `n`: number of coin toss trials.
3. `verbose`: boolean indicating whether to output the estimated parameter values per iteration.

To train on `data`, call

```
estimates = EM_coin.train(data)
```

where `data` is numpy array of coin toss sets, with each value indicating number of heads from one set of 10 tosses. We simulate our data with the following parameter values.

+ $\phi = 0.75$
+ $\theta_{A} = 0.25$
+ $\theta_{B} = 0.60$

```python
import numpy as np

phi = 0.75
thetaA = 0.25
thetaB = 0.60

m = 500

numA = np.random.binomial(m, phi)
numB = m - numA

# generate coin tosses from coin A
tossesA = np.array([np.random.binomial(10, thetaA) for _ in range(numA)])
tossesB = np.array([np.random.binomial(10, thetaB) for _ in range(numB)])

tosses = np.concatenate((tossesA, tossesB))
```

Below is the algorithm implementation.


```python
class EM_coin(object): 
    # Set initial coin bias guesses
    def __init__(self, initial=(0.4, 0.6, 0.500), n = 10, convergence=0.001, 
                 verbose=False):
        self.convergence = convergence
        self.verbose = verbose
        self.n = n
        self.Q = None
        self.parameters = {"thetaA": initial[0], "thetaB": initial[1], "phi": initial[2]}
    
    def likelihood(self, n, p, k):
        return (p**k)*(1-p)**(n-k)
    
    def expectation(self, data):
        lA = [self.likelihood(self.n, self.parameters["thetaA"], x) * self.parameters["phi"] for x in data]
        lB = [self.likelihood(self.n, self.parameters["thetaB"], x) * (1 - self.parameters["phi"]) for x in data]
        self.Q = np.array([lA, lB], dtype = float).T
        total = np.sum(self.Q, axis=1, keepdims=True)
        self.Q[:, [0]] = self.Q[:, [0]] / total
        self.Q[:, [1]] = 1 - self.Q[:, [0]]
    
    def maximization(self, data):
        thetaA_n= np.sum(np.squeeze(self.Q[:, [0]]) * data)
        thetaA_d = self.n * np.sum(self.Q[:, [0]])
        self.parameters["thetaA"] = thetaA_n / thetaA_d
        
        thetaB_n= np.sum(np.squeeze(self.Q[:, [1]]) * data)
        thetaB_d = self.n * np.sum(self.Q[:, [1]])
        self.parameters["thetaB"] = thetaB_n / thetaB_d
        
        totals =  np.sum(self.Q, axis=1, keepdims = True)
        phi = np.sum(self.Q[:, [0]]) / np.sum(totals)
        self.parameters["phi"] = phi
        
    def difference(self, oldParameters):
        return abs(self.parameters["thetaA"] - oldParameters["thetaA"]) +\
                abs(self.parameters["thetaB"] - oldParameters["thetaB"]) +\
                abs(self.parameters["phi"] - oldParameters["phi"])
    
    def train(self, data):
        oldParameters = {"thetaA": 0, "thetaB": 0, "phi": 0}
        i = 1
        
        while self.difference(oldParameters) > self.convergence:
            oldParameters["thetaA"] = self.parameters["thetaA"]
            oldParameters["thetaB"] = self.parameters["thetaB"]
            oldParameters["phi"] = self.parameters["phi"]
            if self.verbose: 
                print(str(i) + ". " + " thetaA: " + str(round(self.parameters["thetaA"], 3)) +\
                      ", thetaB: " + str(round(self.parameters["thetaB"], 3)) +\
                      ", phi: " + str(round(self.parameters["phi"], 3)))
            self.expectation(data)
            self.maximization(data)
            i += 1

        return self.parameters
```

Now, let us create our dataset and train to estimate the coin biases:


```python
algorithmCoin = EM_coin(verbose=True)
estimates = algorithmCoin.train(tosses)
print("Estimates: thetaA =", str(round(estimates["thetaA"], 3)), 
      "; thetaB =", str(round(estimates["thetaB"], 3)), 
      "; phi =", str(round(estimates["phi"], 3)))
```

    1.  thetaA: 0.4, thetaB: 0.6, phi: 0.5
    2.  thetaA: 0.258, thetaB: 0.522, phi: 0.727
    3.  thetaA: 0.245, thetaB: 0.55, phi: 0.723
    4.  thetaA: 0.241, thetaB: 0.561, phi: 0.724
    5.  thetaA: 0.241, thetaB: 0.566, phi: 0.727
    6.  thetaA: 0.241, thetaB: 0.569, phi: 0.73
    7.  thetaA: 0.242, thetaB: 0.571, phi: 0.733
    8.  thetaA: 0.242, thetaB: 0.572, phi: 0.736
    9.  thetaA: 0.243, thetaB: 0.574, phi: 0.738
    10.  thetaA: 0.243, thetaB: 0.575, phi: 0.74
    11.  thetaA: 0.244, thetaB: 0.576, phi: 0.742
    12.  thetaA: 0.244, thetaB: 0.578, phi: 0.744
    13.  thetaA: 0.245, thetaB: 0.578, phi: 0.745
    14.  thetaA: 0.245, thetaB: 0.579, phi: 0.747
    15.  thetaA: 0.245, thetaB: 0.58, phi: 0.748
    16.  thetaA: 0.245, thetaB: 0.581, phi: 0.749
    17.  thetaA: 0.246, thetaB: 0.581, phi: 0.75
    18.  thetaA: 0.246, thetaB: 0.582, phi: 0.75
    19.  thetaA: 0.246, thetaB: 0.582, phi: 0.751
    20.  thetaA: 0.246, thetaB: 0.583, phi: 0.752
    21.  thetaA: 0.246, thetaB: 0.583, phi: 0.752
    Estimates: thetaA = 0.246 ; thetaB = 0.583 ; phi = 0.753


The estimated parameters are $$\phi = 0.753$$, $$\theta_{A} = 0.246$$, and $$\theta_{B} = 0.583$$, which agrees quite well with the true parameters.

## 3.2 ABO blood group allele frequency estimation

The following example is based on that provided by UC Berkeley's STATC245C Computational Statistics course. ABO blood groups are characterized by genotypes at the ABO locus, which has alleles A, B, and O. The mapping between unphased genotypes to blood group phenotype is provided by the following table.

| Genotype | Phenotype |
|:-:|:-:|
|   AO  | A |
|   AA  | A |
|   BO  | B |
|   BB  | B |
|   OO  | O |
|   AB  | AB |

We observe phenotypes, and we wish to estimate the frequencies of alleles A, B, and O. Under Hardy-Weinberg equilibrium at the ABO locus, maternal and paternal alleles are independent, so unphased genotype frequencies are products of allele frequencies. The unobserved complete data is $$X = \begin{bmatrix}X_{AA} & X_{AO} & X_{BO} & X_{BB} & X_{AB} & X_{OO}\end{bmatrix}^{\top}$$, the unphased genotypes. The observed incomplete data is $$Y = \begin{bmatrix}Y_{A} & Y_{B} & Y_{AB} & Y_{O}\end{bmatrix}^{\top}$$, the ABO phenotype. The statistical model for $$X$$ is 

$$
X \sim \text{multinomial}(n, (\pi_{A}^{2}, 2\pi_{A}\pi_{O}, \pi_{B}^{2}, 2\pi_{B}\pi_{O}, 2\pi_{A}\pi_{B}, \pi_{O}^{2})),
$$

and the model for $$Y$$ is

$$
Y \sim \text{multinomial}(n, (\pi_{A}^{2} + 2\pi_{A}\pi_{O}, \pi_{B}^{2} + 2\pi_{B}\pi_{O}, 2\pi_{A}\pi_{B}, \pi_{O}^{2})).
$$

We wish to estimate allele frequencies $$\pi = \begin{bmatrix}\pi_{A} & \pi_{B} & \pi_{O}\end{bmatrix}^{\top}$$. The log-likelihood of $$\pi$$ with complete data is

$$
\ell(\pi) \propto 2X_{AA}\log(\pi_{A}) + X_{AO}\log(2\pi_{A}\pi_{O}) + X_{BO}\log(2\pi_{B}\pi_{O}) + 2X_{BB}\log\pi_{B} + 2X_{OO}\pi_{O} + X_{AB}\log\pi_{AB},
$$

which, when can be maximized with respect to $$\pi$$ under the constraint $$\pi_{A} + \pi_{B} + \pi_{O} = 1$$. For instance, application of Lagrange multipliers yields 

$$\hat{\pi}_{A} = \frac{2X_{AA} + X_{AO} + X_{AB}}{2(X_{AA} + X_{AO} + X_{BB} + X_{BO} + X_{AB} + X_{OO})} = \frac{2X_{AA} + X_{AO} + X_{AB}}{2n}$$

In the observed incomplete data case, the log-likelihood is

$$\ell(\pi) \propto Y_{A}\log(\pi_{A}^{2} + 2\pi_{A}\pi_{O}) + Y_{B}\log(\pi_{B}^{2} + 2\pi_{B}\pi_{O}) + Y_{AB}\log(2\pi_{A}\pi_{B}) + 2Y_{O}\log\pi_{O}.$$

Additionally, we know the relationship between $$Y$$ and $$X$$ as follows

+ $Y_{A} = X_{AA} + X_{AO}$
+ $Y_{B} = X_{BB} + X_{BO}$
+ $Y_{AB} = X_{AB}$
+ $Y_{OO} = X_{OO}$

### 3.2.1 Expectation

Recall that in the expectation step, we set $Q(x) = p(x \lvert y; \pi)$. Then, our log-likelihood becomes

$$
\begin{align*}
\ell(\pi) &= \log p(y; \pi)\\
&= \log \sum_{x}p(y, x; \pi)\\
&= \log \sum_{x}\frac{p(y, x; \pi)Q(x)}{Q(x)}\\
&\geq \sum_{x}Q(x)\log\frac{p(y, x; \pi)}{Q(x)}\\
&= \mathbb{E}_{x \sim Q}\left[\log \frac{p(y, x; \pi)}{Q(x)}\right]\\
&= \mathbb{E}_{x \sim Q}\left[\log p(y; \pi)\right],
\end{align*}
$$

where in the last line, we use the identity $$Q(x) = \frac{p(y, x; \pi)}{p(y; \pi)}$$. Expressing $\ell(\pi)$ only in terms containing $\pi$ and expressing $$Y$$ in terms of $$X$$

$$
\begin{align*}
\ell(\pi) &\propto \mathbb{E}_{x \sim Q}\left[(x_{AA} + x_{AO})\log(\pi_{A}^{2} + 2\pi_{A}\pi_{O}) + (x_{BB} + x_{BO})\log(\pi_{B}^{2} + 2\pi_{B}\pi_{O}) + x_{AB}\log(2\pi_{B}\pi_{A}) + 2x_{OO}\log\pi_{O}\right]\\
&= \log(\pi_{A}^{2} + 2\pi_{A}\pi_{O})(\mathbb{E}[x_{AA} | y_{A}] + \mathbb{E}[x_{AO} | y_{A}]) + \log(\pi_{B}^{2} + 2\pi_{B}\pi_{O})(\mathbb{E}[x_{BB} | y_{B}] + \mathbb{E}[x_{BO} | y_{B}]) + \\
&\quad\mathbb{E}[x_{AB} | y_{AB}]\log(2\pi_{B}\pi_{A}) + 2\mathbb{E}[x_{OO} | y_{OO}]\log\pi_{O}.\\
\end{align*}
$$

For above, we also apply independence assumptions between terms in $X$ and terms in $Y$. For example, $\mathbb{E}[x_{AA} \lvert y_{A}, y_{B}, y_{AB}, y_{O}] = \mathbb{E}[x_{AA} \lvert y_{A}]$ because $X_{AA}$ only has a dependency on $Y_{A}$. It remains to find the expectation terms. For a multinomial distribution, the conditional distribution is

$$
\begin{align*}
p(X_{k} = x | X_{k} + X_{k^{\prime}} = y) &= \frac{p(X_{k} = x, X_{k^{\prime}} = y - x)}{p(X_{k} + X_{k^{\prime}} = y)}\\
&= \frac{\frac{n!}{x!(y - x)!(n - y)!}\pi_{k}^{x}\pi_{k^{\prime}}^{y - x}(1 - \pi_{k} - \pi_{k^{\prime}})^{n - y}}{\frac{n!}{y!(n - y)!}(\pi_{k} + \pi_{k^{\prime}})^{y}(1 - \pi_{k} - \pi_{k^{\prime}})^{n - y}}\\
&= \frac{y!}{x!(y - x)!}\left(\frac{\pi_{k}}{\pi_{k} + \pi_{k^{\prime}}}\right)^{x}\left(\frac{\pi_{k^{\prime}}}{\pi_{k} + \pi_{k^{\prime}}}\right)^{y - x}\\
&\implies X_{k} | X_{k} + X_{k^{\prime}} = y \sim \text{binomial}\left(y, \frac{\pi_{k}}{\pi_{k} + \pi_{k^{\prime}}}\right),
\end{align*}
$$

which means $$\mathbb{E}[X_{k} \lvert X_{k} + X_{k^{\prime}} = y] = y\frac{\pi_{k}}{\pi_{k} + \pi_{k^{\prime}}}$$. Applying this expectation we get

$$
\begin{align*}
\ell(\pi) &\propto y_{A}\log(\pi_{A}^{2} + 2\pi_{A}\pi_{O}) + y_{B}\log(\pi_{B}^{2} + 2\pi_{B}\pi_{O}) + y_{AB}\log(2\pi_{B}\pi_{A}) + 2y_{OO}\log \pi_{O}.
\end{align*}
$$


### 3.2.2 Maximization

We solve the following constrained optimization problem

$$
\begin{align*}
\max_{\pi}\:&\ell(\pi)\\
\sum_{i \in \{A, B, O\}}\pi_{i} &= 1\\
\end{align*}
$$

with method of Lagrangian multipliers, which leads to solving the following system of equations:

$$
\begin{align*}
\frac{\partial \ell(\pi)}{\partial \pi_{A}} &= 2y_{A}\frac{\pi_{A} + \pi_{O}}{\pi_{A}^{2} + 2\pi_{A}\pi_{O}} + \frac{y_{AB}}{\pi_{A}} = \lambda\\
\frac{\partial \ell(\pi)}{\partial \pi_{B}} &= 2y_{B}\frac{\pi_{B} + \pi_{O}}{\pi_{B}^{2} + 2\pi_{B}\pi_{O}} +  \frac{y_{AB}}{\pi_{B}} = \lambda\\
\frac{\partial \ell(\pi)}{\partial \pi_{O}} &= \frac{2y_{A}\pi_{A}}{\pi_{A}^{2} + 2\pi_{A}\pi_{O}} + \frac{2y_{B}\pi_{B}}{\pi_{B}^{2} + 2\pi_{B}\pi_{O}} + \frac{2y_{O}}{\pi_{O}} = \lambda\\
\sum_{i \in \{A, B, O\}}\pi_{i} &= 1
\end{align*}
$$

Multiply the first three equations by $$\pi_{A}$$, $$\pi_{B}$$, and $$\pi_{O}$$ respectively, to get

$$
\begin{align*}
\pi_{A} &= \frac{2y_{A}\frac{\pi_{A}^{2}}{\pi_{A}^{2} + 2\pi_{A}\pi_{O}} + y_{A}\frac{2\pi_{A}\pi_{O}}{\pi_{A}^{2} + 2\pi_{A}\pi_{O}} + y_{AB}}{\lambda}\\
\pi_{B} &= \frac{2y_{B}\frac{\pi_{B}^{2}}{\pi_{B}^{2} + 2\pi_{B}\pi_{O}} + y_{B}\frac{2\pi_{B}\pi_{O}}{\pi_{B}^{2} + 2\pi_{B}\pi_{O}} + y_{AB}}{\lambda}\\
\pi_{O} &= \frac{y_{A}\frac{2\pi_{A}\pi_{O}}{\pi_{A}^{2} + 2\pi_{A}\pi_{O}} + y_{B}\frac{2\pi_{B}\pi_{O}}{\pi_{B}^{2} + 2\pi_{B}\pi_{O}} + 2y_{O}}{\lambda}.
\end{align*}
$$

From $$\pi_{A} + \pi_{B} + \pi_{O} = 1$$, we multiply both sides by $\lambda$ to get $$\lambda = 2n$$. Next, on the right side notice the numerator terms contain expectations already calculated in the expectation step. Thus, simply substitute those values to get

$$
\begin{align*}
\hat{\pi}_{A} &= \frac{2\mathbb{E}[X_{AA} | Y_{A}] + \mathbb{E}[X_{AO} | Y_{A}] + \mathbb{E}[X_{AB} | Y_{AB}]}{2n}\\
\hat{\pi}_{B} &= \frac{2\mathbb{E}[X_{BB} | Y_{B}] + \mathbb{E}[X_{BO} | Y_{B}] + \mathbb{E}[X_{AB} | Y_{AB}]}{2n}\\
\hat{\pi}_{O} &= \frac{\mathbb{E}[X_{AO}|Y_{A}] + \mathbb{E}[X_{BO} | Y_{B}] + 2\mathbb{E}[X_{OO} | Y_{O}]}{2n}.
\end{align*}
$$


### 3.2.3 Implementation

Now we can implement the EM algorithm to estimate allele frequencies. We will first implement the EM algorithm class with the following usage:

```
EM_ABO(initial=(0.500, 0.500, 0.500), convergence=0.001, verbose=False)
```

where

1. `initial`: tuple of size three indicating initial guesses to the coin biases and coin probabilities.
2. `convergence`: threshold of sum of differences in estimated parameters between iterations, below which the EM algorithm stops.
3. `verbose`: boolean indicating whether to output the estimated parameter values per iteration.

To train on `data`, call

```
estimates = EM_ABO.train(data)
```

where `data` is numpy array of observed phenotype $$Y = \begin{bmatrix}Y_{A} & Y_{B} & Y_{AB} & Y_{O}\end{bmatrix}^{\top}$$. We take phenotype counts from Table III from Clarke $$\textit{et al}.$$ [3]. The data is reproduced below:

| Phenotype | Count |
|:-:|:-:|
|  A  | 186 |
|  B  | 38 |
|  AB  | 13 |
|  O  | 284 |



```python
class EM_ABO(object): 
    # Set initial coin bias guesses
    def __init__(self, initial=(1/3, 1/3, 1/3), convergence=0.001, verbose=False):
        self.convergence = convergence
        self.verbose = verbose
        self.parameters = {"pA": initial[0], "pB": initial[1], "pO": initial[2]}
        self.expectations = np.empty((6,))
        self.n = None
    
    def expectation(self, data):
        denom1 = self.parameters["pA"]**2 + 2 * self.parameters["pA"] * self.parameters["pO"]
        denom2 = self.parameters["pB"]**2 + 2 * self.parameters["pB"] * self.parameters["pO"]
        self.expectations[0] = data[0] * (self.parameters["pA"]**2 / denom1)
        self.expectations[1] = data[0] * ((2 * self.parameters["pA"] * self.parameters["pO"]) / denom1)
        self.expectations[2] = data[1] * (self.parameters["pB"]**2 / denom2)
        self.expectations[3] = data[1] * ((2 * self.parameters["pB"] * self.parameters["pO"]) / denom2)
        self.expectations[4] = data[2]
        self.expectations[5] = data[3]
    
    def maximization(self, data):
        self.parameters["pA"] = (2 * self.expectations[0] + self.expectations[1] + self.expectations[4]) / (2 * self.n)
        self.parameters["pB"] = (2 * self.expectations[2] + self.expectations[3] + self.expectations[4]) / (2 * self.n)
        self.parameters["pO"] = (self.expectations[1] + self.expectations[3] + 2 * self.expectations[5]) / (2 * self.n)
        
    def difference(self, oldParameters):
        return abs(self.parameters["pA"] - oldParameters["pA"]) +\
                abs(self.parameters["pB"] - oldParameters["pB"]) +\
                abs(self.parameters["pO"] - oldParameters["pO"])
    
    def train(self, data):
        self.n = np.sum(data)
        oldParameters = {"pA": 0, "pB": 0, "pO": 0}
        i = 1
        
        while self.difference(oldParameters) > self.convergence:
            oldParameters["pA"] = self.parameters["pA"]
            oldParameters["pB"] = self.parameters["pB"]
            oldParameters["pO"] = self.parameters["pO"]
            if self.verbose: 
                print(str(i) + ". " + " pA: " + str(round(self.parameters["pA"], 3)) +\
                      ", pB: " + str(round(self.parameters["pB"], 3)) +\
                      ", pO: " + str(round(self.parameters["pO"], 3)))
            self.expectation(data)
            self.maximization(data)
            i += 1

        return self.parameters
```

Run the algorithm on data


```python
data = np.array([186, 38, 13, 284])
em_abo = EM_ABO(verbose=True)
estimates = em_abo.train(data)
```

    1.  pA: 0.333, pB: 0.333, pO: 0.333
    2.  pA: 0.25, pB: 0.061, pO: 0.688
    3.  pA: 0.218, pB: 0.05, pO: 0.731
    4.  pA: 0.214, pB: 0.05, pO: 0.736
    5.  pA: 0.214, pB: 0.05, pO: 0.736


The estimated allele frequencies are $$\pi_{A} = 0.214$$, $$p_{B} = 0.05$$, and $$p_{O} = 0.736$$. 

## Reference
1. Ng, Andrew. "CS229 Lecture notes." CS229 Lecture notes 1.1 (2000): 1-3.
2. Do, Chuong B., and Serafim Batzoglou. "What is the expectation maximization algorithm?." Nature biotechnology 26.8 (2008): 897-899.
3. Clarke, C. A., et al. "Secretion of blood group antigens and peptic ulcer." British medical journal 1.5122 (1959): 603.
