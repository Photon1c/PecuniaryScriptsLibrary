<p align="center">  
<img src="https://github.com/Photon1c/PecuniaryScriptsLibrary/blob/main/inputs/logo.png?raw=true" alt="Logo"/>
</p>  
    
# Pecuniary Scripts Library

A collection of financial scripts to speed up batch processing input lists through screeners and visualizer output generators.

The sequence of steps that these scripts adhere to is as follows:

-Retrieve a list of stock tickers.  
-Iterate them through screening functions.  
-Update list with output from screening functions to generate charts.  

Check back later as material for this project is released.


```mermaid
stateDiagram-v2
Inputs --> Screeners
Screeners --> Charts
Charts --> Inputs
```
<details>
<summary>Updates</summary>  

# Update 3.3.2025 - 💹

A new tool, the [Metric Visualizer](/scripts/Financial-Metric-Visualizer.ipynb) is now available.  It computes the Sharpe Ratio, Annualized Return, and Volatility to then geneate a chart.

# Update 2.24.2025 - ✏️🗒📊

The [Tangency Portfolio Advanced Report Generator](/scripts/TangencyPortfolio-Advanced-Analysis.py) is now available.  

# Update 2.22.25 -💹👀📓

The [Option AI Payoff Diagram Generator](/inputs/OptionPayoff-AI-Creator.ipynb) is now available. Also, check out the [Stock Candle Wick Analyzers](https://github.com/Photon1c/StockCandleWickAnalyzers) repository for examples of the workflow mentioned on the 2.20.25 update.

# Update 2.20.2025 - 🧠🌠⏲️

The current project workflow is under development, check back for updates:

![AI Stock Vision Flow](/inputs/aistockvision.png)
  
# Update 2.19.2025 - 💻👁️📊

The [Trading View - Chart Extractor](scripts/TradingView-ChartExtractor.ipynb) is a useful image saving script to collect stock charts that LLMs can use for vision analysis.  


# Advanced Screening Added 2.9.2025 ⚗️🔎  

A hypothetical portfolio that contains a given list of tickers may be sorted in the following manner so as to decide which positions to close and which to keep. This is for educational purposes only and is not financial advice, the concepts here are meant to build upon existing ones and branch on to new ones. The [following script](https://github.com/Photon1c/PecuniaryScriptsLibrary/blob/main/scripts/advanced_screener_portfolio_manager.py) uses the logic in the flow diagram below: 

```mermaid
flowchart TD
    A[Start] --> B{Evaluate Stock Performance}
    B -- Gain > 10% --> C{Consider for Gainers}
    B -- Loss > 5% --> D{Consider for Losers}
    
    C -- Short-Term Strategy --> G("Gainers_close: Sell and take profits")
    C -- Long-Term Strategy --> H("Gainers_keep: Hold for potential growth")
    
    D -- Loss Potential Recoverable --> I("Losers_keep: Hold and wait for improvement")
    D -- Loss Unlikely to Recover --> J("Losers_close: Sell and cut losses")
    
    B -- No Significant Change --> K["Monitor and Re-evaluate Later"]
```
</details>

