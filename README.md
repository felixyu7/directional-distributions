## Directional Distributions - Pytorch

Implementation of directional (spherical) probability distributions for PyTorch, with training loss functions and evaluation utilities. Useful for predicting directions on the unit sphere S² with calibrated uncertainty.

Includes the <a href="https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution">von Mises-Fisher</a>, <a href="https://doi.org/10.1007/s11222-017-9756-4">Isotropic Angular Gaussian</a>, and <a href="https://doi.org/10.1007/s11222-017-9756-4">Elliptically Symmetric Angular Gaussian</a> distributions.

## Install

```bash
$ pip install directional-distributions
```

## Usage

### Von Mises-Fisher

```python
import torch
from directional_distributions import von_mises_fisher_loss, VMF

# your network predicts [B, 4]: 3 direction + 1 concentration (κ)

pred = model(x)           # (2, 4)
true = target_directions  # (2, 3) unit vectors

# training

loss = von_mises_fisher_loss(pred, true)
loss.backward()

# evaluation

dist = VMF(pred)

dist.mean_direction  # (2, 3) unit vectors
dist.kappa           # (2,)   concentration

dist.log_pdf(points) # (2, N) log-density at N points on S²
dist.pdf(points)     # (2, N)

fig, ax = dist.plot_mollweide()
```

### Isotropic Angular Gaussian

```python
import torch
from directional_distributions import iag_nll_loss, IAG

# your network predicts [B, 3]: the mean vector μ
# direction = μ/||μ||, concentration = ||μ||

pred = model(x)           # (2, 3)
true = target_directions  # (2, 3) unit vectors

# training

loss = iag_nll_loss(pred, true)
loss.backward()

# evaluation

dist = IAG(pred)

dist.mean_direction  # (2, 3) unit vectors
dist.concentration   # (2,)   ||μ||

dist.log_pdf(points) # (2, N)

fig, ax = dist.plot_mollweide()
```

### Elliptically Symmetric Angular Gaussian

```python
import torch
from directional_distributions import esag_nll_loss, ESAG

# your network predicts [B, 5]: 3 mean vector μ + 2 shape parameters γ
# γ = (0, 0) recovers the isotropic case (IAG)

pred = model(x)           # (2, 5)
true = target_directions  # (2, 3) unit vectors

# training

loss = esag_nll_loss(pred, true)
loss.backward()

# evaluation

dist = ESAG(pred)

dist.mean_direction  # (2, 3) unit vectors
dist.concentration   # (2,)   ||μ||
dist.gamma           # (2, 2) ellipticity parameters

dist.log_pdf(points) # (2, N)

fig, ax = dist.plot_mollweide()
```

### Shared Utilities

```python
from directional_distributions import make_grid, plot_mollweide

# generate a grid of points on S²
grid = make_grid(n_lat=181, n_lon=360)

grid.points  # (N, 3) unit vectors
grid.lon     # (N,)   longitude in radians
grid.lat     # (N,)   latitude in radians
grid.shape   # (181, 360)

# plot any scalar field on the sphere
fig, ax = plot_mollweide(grid, values_2d)
```


## Citations

```bibtex
@article{paine2018elliptically,
  title   = {An elliptically symmetric angular Gaussian distribution},
  author  = {Paine, P.J. and Preston, S.P. and Tsagris, M. and Wood, A.T.A.},
  journal = {Statistics and Computing},
  volume  = {28},
  pages   = {689--697},
  year    = {2018}
}
```

```bibtex
@book{mardia2000directional,
  title     = {Directional Statistics},
  author    = {Mardia, K.V. and Jupp, P.E.},
  publisher = {John Wiley \& Sons},
  year      = {2000}
}
```
