 python testB.py
Most recent date for spy: 12_22_2025
>>> Markov Blanket-Driven Option Pricing: Predictive Model

>>> Loading Option Data
Most recent date for spy: 12_22_2025
Latest SPY price: $684.83 (as of 2025-12-22 00:00:00)
Loaded 1093 option contracts for SPY
Stock price: $684.83

>>> Preparing Features

>>> Training Models
Training predictive models...
Model Performance (Calls):
  Classical Model - MAE: $0.57, R²: 1.000
  Full Model (MB)  - MAE: $0.80, R²: 0.999
  Improvement: -40.2% reduction in MAE

>>> Premium Decomposition

Decomposing Premium into Classical vs Extra Components...       
     Premium Decomposition Analysis      
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric                       ┃  Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Mean Actual Premium          │ $33.66 │
│ Mean Classical Component     │ $33.69 │
│ Mean Extra Component (Delta) │  $0.01 │
│ Extra % of Total             │   0.0% │
│ Std of Extra Component       │  $0.49 │
└──────────────────────────────┴────────┘

>>> Skew Attribution Analysis

Analyzing Skew Attribution...
                   Skew Attribution Analysis
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓┃ Component          ┃ Coefficient (Beta) ┃ Interpretation     ┃┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩│ Intercept (Beta_0) │             0.0111 │ Base skew level    ││ Volatility         │            -0.0031 │ Pure volatility    ││ (Beta_1)           │                    │ effect             ││ Trading Volume     │            -0.0014 │ Liquidity flow     ││ (Beta_2)           │                    │ effect             ││ News/Sentiment     │            -0.0031 │ Sentiment shock    ││ (Beta_3)           │                    │ effect             ││                    │                    │                    ││ Model R²           │              0.031 │ Explained variance │└────────────────────┴────────────────────┴────────────────────┘
Interpretation:
  • Volume contributes 44.9% relative to volatility
  • News/sentiment contributes 100.0% relative to volatility    

Generating visualizations...
Visualization saved: 
..\output\markov_blanket_predictions_20251222_233845.png        

================================================================================
╭────────────────────────── Summary ───────────────────────────╮│ ANALYSIS COMPLETE!                                           ││                                                              ││ The Markov blanket-driven approach has been successfully     ││ applied to option pricing.                                   ││ Key findings:                                                ││   • Full model improves prediction by -40.2%                 ││   • Extra component accounts for 0.0% of premium             ││   • Skew model explains 3.1% of variance                     ││                                                              ││ This demonstrates how Markov blanket features capture market ││ inefficiencies                                               ││ beyond traditional Black-Scholes assumptions.                │╰──────────────────────────────────────────────────────────────╯