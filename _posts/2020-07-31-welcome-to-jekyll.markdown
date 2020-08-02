---
layout: post
title:  "Welcome to Jekyll!"
date:   2020-07-31 23:53:04 -0700
categories: jekyll update
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# 1. Motivation


The standard maximum likelihood estimation (MLE) problem solves for

$$\theta^{*} = \arg \max_{\theta}\ell(\theta),$$

the parameter that maximizes the log-likelihood of observed data $$\ell(\theta)$$, given a statistical model. However, in incomplete data scenarios with unobserved latent variable $Z$, simultaneously solving for $z, \theta$ to maximize the log-likelihood $\ell(\theta, z)$ can be impossible. Conversely, if $z$ were observed, the estimation problem would be easily solvable. 

The expectation-maximization (EM) algorithm is a method to solve for a local maximum likelihood estimate of $\theta$ numerically in incomplete data scenarios, alternating maximization between the two sets of unknowns, keeping the other set fixed. This idea is also known as coordinate ascent. 

# 2. Algorithm

The following presentation is largely based on the notes written by Andrew Ng [1]. Given a dataset of $\{x^{(1)},...,x^{(m)}\}$ of $m$ independent samples, the log-likelihood is given by

$$\ell(\theta) = \sum_{i = 1}^{m}\log p(x^{(i)}; \theta) = \sum_{i = 1}^{m}\log \sum_{z}p(x^{(i)}, z^{(i)}; \theta),$$

where $Z$ is an unknown discrete random variable ($z$'s are outcome values). Now for any distribution $z^{(i)} \sim Q_{i}$ (i.e. $Q_{i}(z^{(i)})$), we can further rewrite $\ell(\theta)$

$$
\begin{align*}
\ell(\theta) &= \sum_{i = 1}^{m}\log \sum_{z^{(i)}}p(x^{(i)}, z^{(i)}; \theta)\\
&= \sum_{i=1}^{m}\log\sum_{z^{(i)}}Q_{i}(z^{(i)})\frac{p(x^{(i)}, z^{(i)};\theta)}{Q_{i}(z^{(i)})}\\
&\geq \sum_{i=1}^{m}\sum_{z^{(i)}}Q_{i}(z^{(i)})\log\frac{p(x^{(i)}, z^{(i)};\theta)}{Q_{i}(z^{(i)})}.
\end{align*}
$$

The inequality above is due to Jensen's inequality applied to concave functions. Jensen's inequality states that for a convex function $f$ and random variable $X$, the following inequality $\mathbb{E}[f(X)] \geq f(\mathbb{E}[X])$ is true. One way to easily recall the direction of inequality is using the variance formula $Var(X) = \mathbb{E}[X^{2}] - \mathbb{E}[X]^{2} \geq 0 \iff \mathbb{E}[X^{2}] \geq \mathbb{E}[X]^{2}$, where $f(x) = x^{2}$ is a convex function. 

Now the expectation and maximization steps can be derived. The expectation step considers current $\theta$ value fixed and sets $Q_{i}(z^{(i)})$ so that the inequality above becomes equality. Start by noticing that

$$\log\sum_{z^{(i)}}\frac{p(z^{(i)}, x^{(i)};\theta)}{Q_{i}(z^{(i)})}Q_{i}(z^{(i)}) = g\Big(\mathbb{E}\Big[\frac{p(z^{(j)}, x^{(i)};\theta)}{Q_{i}(z^{(i)})}\Big]\Big)$$

and

$$\sum_{z^{(i)}}Q_{i}(z^{(i)})\log\frac{p(z^{(i)}, x^{(i)};\theta)}{Q_{i}(z^{(i)})} = 
\mathbb{E}\Big[g\left(\frac{p(z^{(i)}, x^{(i)};\theta)}{Q_{i}(z^{(i)})}\right)\Big],$$

which are summation terms to the left and right of the inequality respectively. Jensen's inequality for concave functions state $g\Big(\mathbb{E}\Big[\frac{p(z^{(j)}, x^{(i)};\theta)}{Q_{i}(z^{(i)})}\Big]\Big) \geq \mathbb{E}\Big[g\left(\frac{p(z^{(i)}, x^{(i)};\theta)}{Q_{i}(z^{(i)})}\right)\Big]$. It is easy to see that in order for equality to be achieved, $\frac{p(z^{(j)}, x^{(i)};\theta)}{Q_{i}(z^{(i)})} = c$ for some constant $c$, since $g(\mathbb{E}[c]) = g(c) = \mathbb{E}[g(c)]$. Choosing $Q_{i}(z^{(i)}) \propto p(x^{(i)}, z^{(i)}; \theta)$ is sufficient to achieve constant value. For $Q_{i}$ to remain a probability distribution, set

$$
\begin{align*}
Q_{i}(z^{(i)}) &= \frac{p(x^{(i)}, z^{(i)};\theta)}{\sum_{z^{(i)}}p(x^{(i)}, z^{(i)};\theta)}\\
&=\frac{p(x^{(i)}, z^{(i)};\theta)}{p(x^{(i)};\theta)}\\
&=p(z^{(i)}|x^{(i)};\theta),
\end{align*}
$$

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



You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.


Jekyll requires blog post files to be named according to the following format:

`YEAR-MONTH-DAY-title.MARKUP`

Where `YEAR` is a four-digit number, `MONTH` and `DAY` are both two-digit numbers, and `MARKUP` is the file extension representing the format used in the file. After that, include the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
