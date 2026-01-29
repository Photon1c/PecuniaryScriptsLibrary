import pandas as pd

df = pd.read_csv("logs/sheep_herding_log.csv")

# Quick sanity check
print("Head:\n", df.head(), "\n")
print("Tail:\n", df.tail(), "\n")
print("Ticker counts:\n", df["ticker"].value_counts(), "\n")
print("Regime counts:\n", df["regime_state"].value_counts(), "\n")

# Simple ratios
df["grazing_ratio"] = df["n_sheep_grazing_near_spot"] / df["n_sheep_total"]
df["pen_ratio"] = df["n_sheep_in_pen"] / df["n_sheep_total"]

# Average behavior by regime
print("Mean ratios by regime:\n",
      df.groupby("regime_state")[["grazing_ratio", "pen_ratio"]].mean(), "\n")

# Markov-style transition matrix for regimes
states = ["range_bound", "breach_up", "breach_down"]
transitions = {(i, j): 0 for i in states for j in states}

# Need at least 2 rows to have a transition
if len(df) > 1:
    prev = df["regime_state"].iloc[0]
    for cur in df["regime_state"].iloc[1:]:
        transitions[(prev, cur)] += 1
        prev = cur

mat = pd.DataFrame(0.0, index=states, columns=states)

for i in states:
    row_total = sum(transitions[(i, j)] for j in states)
    if row_total > 0:
        for j in states:
            mat.loc[i, j] = transitions[(i, j)] / row_total

print("Transition matrix:\n", mat)
