# Multiple Linear Regression — Knowledge Base (Q&A)

Interview-style questions and answers for **Day 50** notebooks and multiple linear regression in general.

Related files:
- [readme.md](./readme.md) — theory summary
- [multiple_linear_regression.ipynb](./multiple_linear_regression.ipynb) — sklearn + 3D plot
- [code-from-scratch.ipynb](./code-from-scratch.ipynb) — normal equation from scratch

---

## How to use this document

| Section | Best for |
|---------|----------|
| **Basic** | First-time learners, quick revision |
| **Intermediate** | Project work, sklearn usage, metrics |
| **Advanced** | Interviews, ML engineer / data scientist roles |
| **FAQ** | Short lookup |
| **Notebook-specific** | Explaining what you built in Day 50 |

---

## Basic concepts

### Q1. What is the difference between simple and multiple linear regression?

**Simple:** one feature.

$$
y = \beta_0 + \beta_1 x
$$

**Multiple:** many features.

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p
$$

Same idea: fit a linear relationship. With two features you get a **plane**; with more features, a **hyperplane**.

---

### Q2. What do intercept and coefficients mean?

- **Intercept (`β₀`)** — predicted target when all features are zero. Often not meaningful if zero is outside the real data range (e.g. body measurements).
- **Coefficient (`βⱼ`)** — expected change in `y` when feature `xⱼ` increases by **1 unit**, **holding other features constant** (ceteris paribus).

In sklearn: `reg.intercept_` and `reg.coef_`.

---

### Q3. What is OLS (Ordinary Least Squares)?

OLS picks coefficients that **minimize the sum of squared residuals**:

$$
\min_{\beta} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

“Least squares” = square each error, then sum. Squaring penalizes large errors more and gives a closed-form solution for linear models.

---

### Q4. What is the prediction formula in code?

For one sample with features `[x₁, x₂, …, xₚ]`:

```python
y_hat = intercept + coef[0]*x1 + coef[1]*x2 + ... 
# or vectorized:
y_hat = X @ coef + intercept
```

This is exactly what `MeraLR.predict()` and sklearn's `LinearRegression.predict()` do.

---

### Q5. Why add a column of ones before fitting?

The intercept is not a feature in raw `X`, but the math is cleaner if we treat it like one:

$$
\tilde{X} = [1 \mid X], \quad \tilde{\beta} = [\beta_0, \beta_1, \ldots, \beta_p]^T
$$

Then `y = X̃ β̃` with no separate intercept term. The notebook splits `betas[0]` → `intercept_` and `betas[1:]` → `coef_`.

---

### Q6. What is a residual?

**Residual** = actual minus predicted:

$$
e_i = y_i - \hat{y}_i
$$

Small residuals → good fit on that point. OLS minimizes the **sum of squared** residuals, not the count of large errors (that's more like MAE).

---

## Intermediate concepts

### Q7. What is the normal equation?

Closed-form OLS solution:

$$
\hat{\beta} = (X^T X)^{-1} X^T y
$$

Used in `code-from-scratch.ipynb` inside `MeraLR.fit()`. Works when `X^T X` is invertible (full rank, no perfect multicollinearity).

---

### Q8. Normal equation vs gradient descent — when to use which?

| Approach | Pros | Cons |
|----------|------|------|
| **Normal equation** | Exact solution in one step; easy to implement | O(n·p²) or worse; slow / unstable when p is huge; needs invertible `X^T X` |
| **Gradient descent** | Scales to large n and p; works with regularization | Iterative; needs tuning (learning rate, epochs) |

Sklearn's `LinearRegression` often uses LAPACK (similar to normal equation) for moderate sizes. Deep learning and huge sparse data use iterative methods.

---

### Q9. What is R² and how do you interpret it?

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

- **1.0** — perfect fit on that set  
- **0.0** — no better than always predicting the mean  
- **Negative** — worse than the mean (common on bad test sets)

**Interview tip:** High train R² + low test R² → overfitting. Always report **test** R² for generalization.

---

### Q10. What are MAE and MSE? How do they differ from R²?

From `multiple_linear_regression.ipynb`:

- **MAE** — mean absolute error; same units as `y`; robust to outliers  
- **MSE** — mean squared error; penalizes big errors more  
- **RMSE** — √MSE; same units as `y`

R² is **scale-free** (fraction of variance explained). MAE/MSE are **absolute** error sizes. Use both: R² for comparison across targets, MAE/RMSE for business interpretation.

---

### Q11. Why use train/test split?

To estimate performance on **unseen** data. Fit on `X_train, y_train`; evaluate on `X_test, y_test`. If you evaluate on training data, you overestimate quality because the model has already seen those points.

Typical split: 80/20 or 70/30. Use `random_state` for reproducibility.

---

### Q12. Does scaling features change R² or the fitted plane?

**Standardization** (zero mean, unit variance) changes **coefficient values** but not predictions or R² if you apply the same transform at predict time.

Scaling matters when:
- Comparing coefficient magnitudes (“which feature matters most?”)
- Using regularization (Ridge/Lasso)
- Features have very different units (age vs income)

OLS without regularization on unscaled data still finds the same best hyperplane in raw feature space.

---

### Q13. What is multicollinearity?

Two or more features are **highly correlated** (or linear combinations of each other).

**Effects:**
- Coefficients become unstable (small data changes → big β swings)
- Hard to interpret individual feature effects
- `X^T X` near singular → normal equation numerically unstable

**Fixes:** drop redundant features, PCA, Ridge regression, or collect more data.

---

### Q14. What are the classical linear regression assumptions?

Often remembered as **LINE** or similar:

1. **Linearity** — E[y | X] is linear in parameters  
2. **Independence** — errors uncorrelated across samples  
3. **Homoscedasticity** — constant error variance  
4. **Normality of errors** — mainly for confidence intervals and p-values  
5. **No perfect multicollinearity** among features  

For **prediction only**, normality matters less. For **inference** (p-values, CIs), it matters more.

---

## Advanced / interview concepts

### Q15. What is the bias–variance tradeoff in linear regression?

- **Underfitting (high bias):** model too simple (e.g. linear when truth is nonlinear) → high error on train and test  
- **Overfitting (high variance):** model memorizes noise → low train error, high test error  

Plain OLS with many features relative to samples tends to overfit. **Regularization** (Ridge, Lasso) adds bias to reduce variance.

---

### Q16. Ridge vs Lasso vs Elastic Net?

All add a penalty to OLS:

| Method | Penalty | Effect |
|--------|---------|--------|
| **Ridge (L2)** | λ Σ βⱼ² | Shrinks coefficients; rarely zeroes them |
| **Lasso (L1)** | λ Σ \|βⱼ\| | Can zero out features → feature selection |
| **Elastic Net** | L1 + L2 mix | Useful when features are correlated |

Interview one-liner: **Ridge** when many small effects; **Lasso** when you want sparsity.

---

### Q17. Why can adjusted R² be better than R²?

R² **always increases** (or stays same) when you add features, even useless ones.

**Adjusted R²** penalizes extra features:

$$
R^2_{adj} = 1 - (1 - R^2)\frac{n-1}{n-p-1}
$$

Use it to compare models with **different numbers of predictors** on the same dataset.

---

### Q18. What is heteroscedasticity?

Error variance **changes** with X (e.g. wider spread for larger predictions). Violates an OLS assumption.

**Symptoms:** funnel shape in residual vs fitted plot.  
**Impact:** coefficients may still be unbiased but standard errors wrong.  
**Fixes:** transform target (log y), weighted least squares, or robust standard errors.

---

### Q19. What is the difference between prediction and inference?

- **Prediction:** minimize error on new `y`; care about MAE, RMSE, R²  
- **Inference:** understand **which features matter** and **how much**; care about p-values, confidence intervals, causality  

Large coefficients don't always mean important features (collinearity, scale). Always ask: “prediction or explanation?”

---

### Q20. Can linear regression prove causation?

**No.** It shows **association** controlling for included features. Causation needs experimental design, domain knowledge, or causal methods (IV, DAGs, RCTs).

Interview answer: “Regression tells us correlation conditional on other variables, not causation unless assumptions for causal inference hold.”

---

### Q21. What happens if p > n (more features than samples)?

`X^T X` is singular — **infinite** many solutions fit training data perfectly. Normal equation fails or is unstable.

**Solutions:** reduce p, regularization (Ridge), or more data.

---

### Q22. How does polynomial regression relate to multiple linear regression?

Features like `x, x², x³` are still **linear in the parameters**:

$$
y = \beta_0 + \beta_1 x + \beta_2 x^2
$$

So it's **multiple linear** regression with engineered features, not a different model class. You can fit curves while still using OLS.

---

### Q23. What is the geometric interpretation of OLS?

OLS finds the point in the **column space of X** closest to **y** (orthogonal projection). Residuals are perpendicular to every feature column. This explains why adding a useless feature still changes other coefficients unless it's orthogonal to existing ones.

---

### Q24. Why use `np.linalg.lstsq` instead of explicit inverse in production?

`np.linalg.inv(X.T @ X) @ X.T @ y` is fine for teaching but **less stable** than:

```python
np.linalg.lstsq(X_aug, y, rcond=None)
```

or sklearn's solver. Explicit matrix inverse amplifies numerical errors when features are correlated.

---

## FAQ (quick answers)

**Is multiple linear regression still used?**  
Yes. Baseline for tabular data, interpretable models, and as a building block inside GLMs, econometrics, and feature pipelines.

**Can it handle categorical features?**  
Not directly. Use **one-hot encoding** (or target encoding with care). Each category becomes a 0/1 column.

**Can it handle missing values?**  
Not in raw form. Impute, drop rows, or use models that handle NaNs.

**Is the intercept always needed?**  
Use intercept unless you **know** the true relationship passes through origin. Sklearn `fit_intercept=True` by default.

**What if the relationship is nonlinear?**  
Transform features (log, polynomial), use splines, GAMs, or nonlinear models (trees, neural nets).

**Why did my scratch model match sklearn?**  
Same OLS objective and full-rank data → same β (up to floating-point noise). Diabetes dataset with 10 features and hundreds of samples is well-conditioned for the normal equation.

**What's a good R² for the diabetes benchmark?**  
~0.44 on a hold-out split is typical for vanilla linear regression on that dataset — not “bad,” just a hard real-world problem.

---

## Notebook-specific Q&A

### Q25. What does `make_regression` do in `multiple_linear_regression.ipynb`?

Creates **synthetic** data with known linear structure:

```python
X, y = make_regression(n_samples=100, n_features=2, n_informative=2, noise=50)
```

Good for learning and 3D plots because you control feature count and noise.

---

### Q26. Why build a meshgrid for the 3D plot?

To draw the **fitted plane** over a grid of `(feature1, feature2)` values:

1. `np.linspace` → axis ranges  
2. `np.meshgrid` → all (x, y) pairs  
3. `final` → shape `(100, 2)` for `lr.predict`  
4. `reshape(10, 10)` → surface heights for Plotly  

**Order matters:** define `final` **before** `lr.predict(final)`.

---

### Q27. What does `MeraLR` teach that sklearn hides?

1. How intercept is folded into the design matrix  
2. The normal equation `(X^T X)^(-1) X^T y`  
3. That `fit` = solve for β, `predict` = apply β  

Same API as sklearn (`fit`, `predict`, `coef_`, `intercept_`) so you can swap implementations and compare R².

---

### Q28. Why use the diabetes dataset in `code-from-scratch.ipynb`?

Real tabular regression benchmark: **442 samples, 10 features**, target is disease progression. More realistic than synthetic 2D data; shows OLS works in higher dimensions.

---

### Q29. Walk through `MeraLR.fit()` in an interview.

1. Insert column of 1s → augmented matrix `X̃`  
2. Compute `X̃^T X̃` and `X̃^T y`  
3. Solve for β = `(X̃^T X̃)^(-1) X̃^T y`  
4. Split β₀ → intercept, β₁…βₚ → coef  
5. Predict with `X @ coef + intercept`

---

### Q30. How would you improve the Day 50 pipeline?

Possible answers (pick what fits the role):

- Feature scaling + Ridge for stability  
- Cross-validation instead of single train/test split  
- Residual plots for linearity / heteroscedasticity  
- Compare MAE, RMSE, adjusted R²  
- Replace explicit inverse with `lstsq` or sklearn  
- Feature selection if multicollinearity suspected  

---

## Common interview traps

| Trap | Correct response |
|------|------------------|
| “Higher R² always means better model” | Only on same test set; can overfit; use test R² and simpler baselines |
| “Largest coefficient = most important feature” | Only valid with scaled features and low collinearity |
| “Linear regression assumes y is normal” | Assumes **errors** are roughly normal for inference, not y itself |
| “We don't need train/test with linear regression” | Always need hold-out or CV to estimate generalization |
| “Normal equation works for any dataset” | Needs n ≥ p+1 and full rank; use regularization or iterative methods otherwise |

---

## One-minute elevator pitch

> Multiple linear regression models the target as a weighted sum of features plus an intercept. We fit it with OLS by minimizing squared errors, either via the normal equation or numerical optimization. We evaluate with train/test splits using R² and error metrics. It's interpretable and fast, but assumes roughly linear relationships and suffers when features are redundant or p is large relative to n. Regularized extensions (Ridge/Lasso) fix many of those issues while keeping the linear structure.

---

## Further reading (concepts only)

- **Gauss–Markov theorem** — OLS is BLUE under classical assumptions  
- **VIF (Variance Inflation Factor)** — detect multicollinearity  
- **Statsmodels OLS** — p-values and summary tables for inference  
- **Generalized Linear Models (GLM)** — linear predictor + non-Gaussian targets  

---

*Last updated for Day 50 — Multiple Linear Regression notebooks.*
