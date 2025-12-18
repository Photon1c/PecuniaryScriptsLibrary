# Major Gamma Related Equations

**Plug 1: Time-window energy**
```math
\mathrm{RV}_{t,W} = \int_{t-W}^{t} r_u^2 \, du \;\;\approx\;\; \sum r_u^2
\qquad
E_{\text{IV}} = \int_{t-W}^{t} \left( \frac{d\sigma_{\text{imp}}}{du} \right)^2 du
```

**Plug 2: Cross-sectional exposures**
```math
\mathrm{GEX}_t = \iint \Gamma(K,\tau) \cdot \text{OI}(K,\tau) \, dK \, d\tau
```

**Plug 3: Spectral band energy**
```math
E_{\text{band}} = \int_{f_1}^{f_2} \text{PSD}(f) \, df
```

**Plug 4: Hazard accumulation**
```math
\Lambda(t) = \int_{t-W}^{t} \lambda(u) \, du,
\qquad
p(t) = 1 - e^{-\Lambda(t)}
```
