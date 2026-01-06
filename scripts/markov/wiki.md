Markov Blanket
In a Bayesian network, the Markov boundary of node A includes its parents, children, and the other parents of all of its children.
In statistics and machine learning, a Markov blanket of a random variable is a set of variables that renders the variable conditionally independent of all other variables in the system. This concept is central in probabilistic graphical models and feature selection. If a Markov blanket is minimal—meaning that no variable in it can be removed without losing this conditional independence—it is called a Markov boundary. Identifying a Markov blanket or boundary allows for efficient inference and helps isolate relevant variables for prediction or causal reasoning. The terms Markov blanket and Markov boundary were coined by Judea Pearl in 1988. A Markov blanket may be derived from the structure of a probabilistic graphical model such as a Bayesian network or Markov random field.
Markov Blanket
A Markov blanket of a random variable $Y$ in a random variable set $\mathcal{S} = \{X_1, \ldots, X_n\}$ is any subset $\mathcal{S}_1$ of $\mathcal{S}$, conditioned on which other variables are independent with $Y$:
$$Y \perp\!\!\!\perp \mathcal{S} \setminus \mathcal{S}_1 \mid \mathcal{S}_1$$
It means that $\mathcal{S}_1$ contains at least all the information one needs to infer $Y$, where the variables in $\mathcal{S} \setminus \mathcal{S}_1$ are redundant.
In general, a given Markov blanket is not unique. Any set in $\mathcal{S}$ that contains a Markov blanket is also a Markov blanket itself. Specifically, $\mathcal{S}$ is a Markov blanket of $Y$ in $\mathcal{S}$.
Example
In a Bayesian network, the Markov blanket of a node consists of its parents, its children, and its children's other parents (i.e., co-parents). Knowing the values of these nodes makes the target node conditionally independent of the rest of the network. In a Markov random field, the Markov blanket of a node is simply its immediate neighbors.
Markov Condition
The concept of a Markov blanket is rooted in the Markov condition, which states that in a probabilistic graphical model, each variable is conditionally independent of its non-descendants given its parents. This condition implies the existence of a minimal separating set — the Markov blanket — that shields a variable from the rest of the network.
For instance, when a person holds an object stationary against gravity, the object’s acceleration is fully determined by its direct causes—namely, the upward force from the hand and the downward gravitational pull. Other variables such as air pressure or temperature are causally irrelevant.
Markov Boundary
A Markov boundary of $Y$ in $\mathcal{S}$ is a subset $\mathcal{S}_2$ of $\mathcal{S}$, such that $\mathcal{S}_2$ itself is a Markov blanket of $Y$, but any proper subset of $\mathcal{S}_2$ is not a Markov blanket of $Y$. In other words, a Markov boundary is a minimal Markov blanket.
The Markov boundary of a node $A$ in a Bayesian network is the set of nodes composed of $A$'s parents, $A$'s children, and $A$'s children's other parents. In a Markov random field, the Markov boundary for a node is the set of its neighboring nodes. In a dependency network, the Markov boundary for a node is the set of its parents.
Uniqueness of Markov Boundary
The Markov boundary always exists. Under some mild conditions, the Markov boundary is unique. However, for most practical and theoretical scenarios multiple Markov boundaries may provide alternative solutions. When there are multiple Markov boundaries, quantities measuring causal effect could fail.
See Also

Andrey Markov
Free energy minimisation
Moral graph
Separation of concerns
Causality
Causal inference

References

Pearl, Judea (1988). Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference. Representation and Reasoning Series. San Mateo, CA: Morgan Kaufmann. ISBN 0-934613-73-7.
Statnikov, Alexander; Lytkin, Nikita I.; Lemeire, Jan; Aliferis, Constantin F. (2013). "Algorithms for discovery of multiple Markov boundaries". Journal of Machine Learning Research. 14: 499–566.
Wang, Yue; Wang, Linbo (2020). "Causal inference in degenerate systems: An impossibility result". Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics: 3383–3392.
Add to chat