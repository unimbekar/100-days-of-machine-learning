# Day 50 — Multiple Linear Regression

Video: https://youtu.be/ashGekqstl8

**Interview Q&A:** [knowledge.md](./knowledge.md) — FAQ and basic/advanced questions on multiple linear regression.

## What is multiple linear regression?

**Simple linear regression** uses one feature:

$$
y = \beta_0 + \beta_1 x
$$

**Multiple linear regression** extends that to many features at once:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p
$$

- $y$ — target (what we predict)
- $x_1, \ldots, x_p$ — input features
- $\beta_0$ — **intercept** (predicted $y$ when all features are zero)
- $\beta_1, \ldots, \beta_p$ — **coefficients** (how much $y$ changes when one feature increases by 1, holding others fixed)

In matrix form, with $n$ samples:

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}
$$

We choose $\boldsymbol{\beta}$ to minimize the sum of squared errors (OLS).

## Normal equation (closed form)

When $\mathbf{X}^T\mathbf{X}$ is invertible, the best-fit coefficients are:

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

That is what `MeraLR.fit()` implements in `code-from-scratch.ipynb`: add a column of ones for the intercept, then apply this formula.

**Sklearn** `LinearRegression` uses the same idea for small/medium datasets (via LAPACK). For very large data, iterative solvers are often used instead.

## Interpreting coefficients

Each $\beta_j$ is the expected change in $y$ for a **one-unit increase** in feature $x_j$, assuming other features stay the same.

- Positive $\beta_j$ → feature pushes predictions up  
- Negative $\beta_j$ → feature pushes predictions down  
- Larger $|\beta_j|$ → stronger linear effect (only meaningful if features are on comparable scales)

## Evaluating the model: R²

**R² (coefficient of determination)** measures how much variance in $y$ the model explains:

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

- $R^2 = 1$ — perfect fit on the data used for scoring  
- $R^2 = 0$ — model is no better than predicting the mean  
- Can be negative on a hold-out set if the model generalizes poorly

Always check **train vs test** R² to spot overfitting.

## Assumptions (linear regression)

Classical linear regression works best when:

1. **Linearity** — relationship between features and target is roughly linear  
2. **Independence** — errors on different samples are not correlated  
3. **Homoscedasticity** — error variance is roughly constant  
4. **No severe multicollinearity** — features are not almost perfectly redundant  
5. **Errors are roughly normal** (mainly for inference/p-values; less critical for pure prediction)

## Notebooks in this folder

| File | Purpose |
|------|---------|
| `multiple_linear_regression.ipynb` | sklearn workflow + 3D visualization |
| `code-from-scratch.ipynb` | Custom `MeraLR` class using the normal equation |

## Quick intuition

Multiple linear regression fits a **hyperplane** in feature space instead of a single line. With two features, that plane is easy to visualize; with ten features (e.g. diabetes dataset), the math is the same — only the dimension is higher.
