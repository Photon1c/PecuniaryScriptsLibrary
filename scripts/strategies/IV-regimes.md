## IV Regime Trading Strategy
```mermaid
flowchart TB
  %% IV Rank (vertical) + IV Percentile (horizontal) quadrant
  %% Left = Low Percentile, Right = High Percentile
  %% Bottom = Low Rank, Top = High Rank
  subgraph Q[IV Context Quadrant — Rank × Percentile]
    direction LR
    subgraph Left[Low IV Percentile]
      direction TB
      A[HIGH Rank + LOW Percentile\nOne-off spike / repricing risk\n→ Be careful fading\n→ Prefer DEFENSIVE / WAIT]:::spike
      B[LOW Rank + LOW Percentile\nVol cheap + complacency\n→ Convexity favored\n→ Prefer LONG GAMMA / DEFINED-RISK]:::cheap
    end
    subgraph Right[High IV Percentile]
      direction TB
      C["HIGH Rank + HIGH Percentile\nVol rich + often this high\n→ Fade favored\n→ Prefer SHORT VOL (defined-risk)"]:::rich
      D[LOW Rank + HIGH Percentile\nPersistent low-vol regime\n→ Carry dominates\n→ Prefer NEUTRAL / SELECTIVE]:::persist
    end
  end
  %% Optional “gates” to link quadrant → action
  C --> F["Fade eligible IF IV slope ≤ 0\nAND not breaking structure"]:::gate
  B --> G["Convexity eligible IF IV slope turns ↑\nOR near hinge (flip/walls)"]:::gate
  A --> H["Defensive: wait for confirmation\nor size down"]:::gate
  D --> I["Neutral: structures depend on view\n(use spreads, avoid paying fat premium)"]:::gate
  classDef rich fill:#1f7a1f,color:#ffffff,stroke:#0b3d0b,stroke-width:2px;
  classDef cheap fill:#1f4aa8,color:#ffffff,stroke:#0b1f4a,stroke-width:2px;
  classDef spike fill:#a85a1f,color:#ffffff,stroke:#4a260b,stroke-width:2px;
  classDef persist fill:#6b6b6b,color:#ffffff,stroke:#2b2b2b,stroke-width:2px;
  classDef gate fill:#111827,color:#ffffff,stroke:#374151,stroke-width:1px;
```
IV Rank & IV Percentile → Fade vs Convexity (One‑Page Study Sheet)
Quick definitions

IV Rank: where today’s IV sits between the 1Y low–high range (0–100).

IV Percentile: % of days in the lookback where IV was lower than today (0–100).

How to use them

Fade IV (short vol) = bet on mean reversion / vol coming down.

Deploy convexity (long vol) = bet on regime change / acceleration.

Mermaid overview diagram

```mermaid
flowchart TB
  A[Inputs] --> B["IV Rank\n0–100\n\"How cheap vs its range?\""]
  A --> C["IV Percentile\n0–100\n\"How often this high?\""]
  A --> D["IV Slope\nΔIV / time\n\"Is vol rising now?\""]
  A --> E["Structure\nSpot vs Flip/Walls\n\"Where are we on the map?\""]
  
  B --> F{"Vol Context\n(Rank × Percentile)"}
  C --> F
  
  F -->|"Low Rank + Low Percentile"| G["Vol is CHEAP & RARELY this low\n→ Complacency risk"]
  F -->|"High Rank + High Percentile"| H["Vol is RICH & OFTEN this high\n→ Mean-reversion likely"]
  F -->|"High Rank + Low Percentile"| I["One-off SPIKE\n→ Repricing risk"]
  F -->|"Low Rank + High Percentile"| J["Persistent LOW-VOL regime\n→ Carry dominates"]
  
  D --> K{"Slope Gate"}
  K -->|"Slope ↑ (positive)"| L["Vol is being BID now\n→ Avoid fading"]
  K -->|"Slope ↓/flat"| M["Vol stable/bleeding\n→ Fade more plausible"]
  
  G --> N[Convexity Bias]
  H --> O[Fade Bias]
  I --> P[Defensive / Wait]
  J --> Q[Neutral / Selective]
  
  N --> R{Structure Gate}
  O --> R
  P --> R
  Q --> R
  
  R -->|"Near Flip/Walls + Break risk"| S["Deploy convexity\n(defined-risk longs, spreads, straddles if justified)"]
  R -->|"Inside pin zone / stable"| T["Fade IV\n(defined-risk short vol: spreads/condors/covered writes)"]
  R -->|"Unclear / conflicting"| U["Abstain / Neutral\n(wait for confirmation)"]
```

## IV Rank × IV Percentile — Fade vs Convexity Matrix

| IV Rank | IV Percentile | Market Interpretation | Strategic Bias |
|--------:|--------------:|----------------------|----------------|
| **Low** | **Low** | Volatility is cheap and rarely this low (complacency regime) | **Deploy convexity** (long gamma, defined-risk longs) |
| **High** | **High** | Volatility is rich and often elevated (mean-reversion regime) | **Fade IV** (defined-risk short vol, carry structures) |
| **High** | **Low** | One-off volatility spike (possible repricing / regime shift) | **Defensive / wait** — fading is dangerous |
| **Low** | **High** | Persistent low-volatility environment (stable carry regime) | **Neutral / selective** — convexity delayed |

> **Rule of thumb:** Fade IV only when *both* Rank and Percentile are high and IV slope is not rising; deploy convexity when volatility is cheap or just beginning to reprice near structural levels.



