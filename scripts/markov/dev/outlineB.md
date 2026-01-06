Potential extensions — This could evolve into a full predictive model: train an ML regressor on just the blanket features to forecast premiums, or compare its performance to Black-Scholes on historical mispricings.

Markov Blanket–Driven Option Pricing: A Causal Feature Selection Approach
Abstract (Draft Excerpt)
Traditional option pricing models, such as the Black-Scholes framework, rely on a fixed set of five inputs—spot price, volatility, risk-free rate, time to expiration, and strike price—under the assumption of lognormal asset dynamics and constant implied volatility. However, real-world option premiums exhibit persistent anomalies including volatility skew, smile, and liquidity premia that are not fully explained by these classical variables. This paper proposes a novel approach that leverages the Markov blanket from probabilistic graphical models to identify the minimal set of variables that render the option premium conditionally independent of all other market factors. By constructing a Bayesian network over key financial variables and computing the Markov blanket of the option premium node, we isolate an optimal, causally motivated feature set that includes not only the classical inputs but also trading volume (as a direct effect) and news/sentiment indicators (as co-parents). This framework enables decomposition of the premium into a “classical” component (explained by Black-Scholes-like inputs) and an “extra” component attributable to market flows and external shocks, offering a principled way to quantify how much of observed skew arises from sentiment-driven demand versus pure volatility.
Core Methodology and Equations
We model the joint distribution of market variables using a directed acyclic graph (DAG) $\mathcal{G} = (V, E)$, where $V$ includes the option premium $P$ and covariates such as spot price $S_t$, implied volatility $\sigma$, risk-free rate $r$, time to expiration $\tau$, strike $K$, trading volume $V$, and a news/sentiment proxy $N$. The Markov blanket $\mathcal{MB}(P)$ of the target node $P$ is defined as the union of its parents $\mathrm{pa}(P)$, children $\mathrm{ch}(P)$, and the co-parents (other parents of its children):
$$\mathcal{MB}(P) = \mathrm{pa}(P) \cup \mathrm{ch}(P) \cup \bigcup_{X \in \mathrm{ch}(P)} \bigl( \mathrm{pa}(X) \setminus \{P\} \bigr)$$
By the Markov condition in the Bayesian network, conditioning on the blanket renders $P$ independent of the remaining variables:
$$P \perp\!\!\!\perp (V \setminus \mathcal{MB}(P)) \;\big|\; \mathcal{MB}(P)$$
where $V$ is the full set of nodes. Empirically, we estimate the blanket via constraint-based structure learning (e.g., PC algorithm) or score-based methods, yielding a minimal set typically comprising the five classical variables plus trading volume and news.
To decompose the premium, we train two predictive models:

$f_{\text{classical}}(S_t, \sigma, r, \tau, K)$ ≈ Black-Scholes implied premium
$f_{\text{full}}(\mathcal{MB}(P))$ ≈ observed premium

The residual $\Delta = f_{\text{full}}(S_t, \sigma, r, \tau, K, V, N) - f_{\text{classical}}(S_t, \sigma, r, \tau, K)$ captures the incremental contribution of volume and news. For skew attribution, we regress the put–call implied volatility difference (skew metric) against the blanket variables:
$$\text{Skew} = \beta_0 + \beta_1 \sigma + \beta_2 V + \beta_3 N + \epsilon$$
The coefficients $\beta_2$ and $\beta_3$ quantify the portion of skew driven by liquidity flows and sentiment shocks, respectively, beyond pure volatility $\sigma$. Future work will extend this to time-series backtesting and machine learning ensembles for out-of-sample premium forecasting.
Sleep well — we can dive into implementation details tomorrow!