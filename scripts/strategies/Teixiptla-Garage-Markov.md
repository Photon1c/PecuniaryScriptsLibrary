# The Teixiptla-Garage-Markov Framework

This document presents a regime-aware trading framework that blends gamma exposure dynamics, implied volatility structure, and a Markov causal perspective to help traders interpret options flow pressure and price behaviour. At its core is the idea that the market’s internal forces — particularly dealer hedging flows — create stability or instability regimes observable via tools like Barchart’s Gamma Exposure (GEX) charts. When price is below the gamma flip point, dealers are typically net short gamma, amplifying volatility and momentum; when above it, dealers are net long gamma, dampening volatility and encouraging range behaviour. By integrating this with IV rank/percentile and time horizon context, traders can derive a permission system that guides whether to deploy convexity (long options / debit spreads), theta-harvesting structures (credit spreads / flies), or to avoid trades altogether. This strategy uses a Markov Blanket analogy to distinguish core drivers (spot, vol, time), boundary conditions (liquidity, news, hedging), and noise — helping clarify when structural signals are meaningful versus when the market is dominated by randomness.

![seal](https://github.com/Photon1c/PecuniaryScriptsLibrary/blob/main/inputs/media/TeixiptlaGarageSeal.png)

```mermaid
flowchart TB
  A[Market Snapshot] --> B[Garage + Teixiptla + Markov Frame]
  B --> C{Dealer Gamma Regime}

  C -->|Spot < Flip| NG[Negative Gamma\nDealers SHORT gamma\nChase price\nVol expands]
  C -->|Spot > Flip| PG[Positive Gamma\nDealers LONG gamma\nLean vs price\nVol compresses]

  NG --> D{IV Regime}
  PG --> D

  D -->|IV Low| IVL[IV Cheap\nConvexity affordable]
  D -->|IV High| IVH[IV Expensive\nConvexity taxed]

  IVL --> E{Time Horizon}
  IVH --> E

  E -->|Enough time| TE[Time Available\nRoom to resolve]
  E -->|Late or expiry| TL[Time Thin\nPin risk]

  TE --> F{Permission Gate}
  TL --> F

  F -->|NG + IVL + TE| OK[DEPLOY CONVEXITY\nBuy options or debit spreads\nMarkov: breakout path likely]
  F -->|PG + TL| CHOP[THETA / FADE\nCredit spreads or flies\nMarkov: reversion attractor]
  F -->|NG + IVH| TAX[WARNING\nConvexity taxed\nPrefer spreads, smaller size]
  F -->|PG + IVL + TE| PIN[WARNING\nDormant convexity\nRange bias unless catalyst]
  F -->|IVH + TL| NO[NO TRADE\nLate + expensive vol\nRitual waste]

  OK --> MB1[Markov Blanket Inside\nSpot Vol Time Strike Rate]
  CHOP --> MB2[Markov Blanket Boundary\nLiquidity News Dealer hedging]
  TAX --> MB2
  PIN --> MB1
  NO --> MB3[Markov Blanket Outside\nNoise field]

```
```mermaid
flowchart TB
  L[LEGEND]

  L --> G[Gamma Regime\nNG = Negative Gamma\nPG = Positive Gamma]

  L --> V[IV Regime\nIV = Implied Volatility\nIVL = Low IV Rank or Percentile\nIVH = High IV Rank or Percentile]

  L --> T[Time Regime\nTE = Time Available\nTL = Time Thin or Late Session]

  L --> P[Permission Outcomes\nOK = Deploy Convexity\nCHOP = Theta or Fade Range\nTAX = Convexity is Expensive\nPIN = Dormant Convexity\nNO = No Trade]

  L --> M[Markov Layers\nInside = Core pricing drivers\nBoundary = Liquidity, news, hedging\nOutside = Noise field]



```
