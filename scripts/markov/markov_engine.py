"""
Markov Engine: Discrete-time and continuous-time Markov chain analysis for regime/gate states.

This module implements:
1. Discrete-time transition matrix P estimation from empirical state sequences
2. Stationary distribution π solving π = πP
3. Continuous-time generator Q and Kolmogorov forward equation dp/dt = pQ
4. Markov risk flags and Kelly modifiers based on π and flow diagnostics

Based on the conceptual framework:
- Markov (1905): P(X_{n+1} = j | X_n = i, ...) = P_ij, with ergodic convergence π = πP
- Kolmogorov (1931): dp/dt = pQ for continuous-time evolution
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from enum import Enum


class RegimeState(Enum):
    """Regime states for the Markov chain."""
    PIN = "PIN"
    EXPRESSIVE = "EXPRESSIVE"
    RUPTURE_CANDIDATE = "RUPTURE_CANDIDATE"
    GARAGE = "GARAGE"
    NEUTRAL = "NEUTRAL"


class GateState(Enum):
    """Gate states for the Markov chain."""
    BLOCK = "BLOCK"
    THROTTLED = "THROTTLED"
    OPEN = "OPEN"


class MarkovRiskFlag(Enum):
    """Risk flags derived from Markov diagnostics."""
    STABLE_PIN = "STABLE_PIN"
    DRIFTING_EXPRESSIVE = "DRIFTING_EXPRESSIVE"
    RUPTURE_DRIFT = "RUPTURE_DRIFT"
    TRANSIENT_UNCERTAIN = "TRANSIENT_UNCERTAIN"


class MarkovEngine:
    """
    Markov engine for regime/gate state analysis.
    
    Treats regime and gate states as a Markov chain, estimates transition probabilities,
    computes stationary distributions, and evolves continuous-time flows.
    """
    
    def __init__(self, regime_states: Optional[List[str]] = None, gate_states: Optional[List[str]] = None):
        """
        Initialize Markov engine with state space.
        
        Args:
            regime_states: List of regime state names (default: all RegimeState values)
            gate_states: List of gate state names (default: all GateState values)
        """
        # Default state spaces
        if regime_states is None:
            regime_states = [s.value for s in RegimeState]
        if gate_states is None:
            gate_states = [s.value for s in GateState]
        
        self.regime_states = regime_states
        self.gate_states = gate_states
        
        # Combined state space: use regime as primary, gate as secondary
        # For simplicity, we'll use regime states as the main chain
        # Gate states can be incorporated later if needed
        self.state_names = regime_states.copy()
        self.n_states = len(self.state_names)
        
        # State index mapping
        self.state_to_idx = {state: idx for idx, state in enumerate(self.state_names)}
        self.idx_to_state = {idx: state for state, idx in self.state_to_idx.items()}
    
    def encode_state(self, regime: str, gate: Optional[str] = None) -> int:
        """
        Encode regime (and optionally gate) into state index.
        
        Args:
            regime: Regime state name
            gate: Optional gate state name (currently unused, for future expansion)
        
        Returns:
            State index
        """
        regime_upper = regime.upper()
        if regime_upper in self.state_to_idx:
            return self.state_to_idx[regime_upper]
        # Fallback: map unknown states to NEUTRAL or first state
        if "NEUTRAL" in self.state_to_idx:
            return self.state_to_idx["NEUTRAL"]
        return 0
    
    def decode_state(self, idx: int) -> str:
        """
        Decode state index to human-readable label.
        
        Args:
            idx: State index
        
        Returns:
            State name
        """
        return self.idx_to_state.get(idx, self.state_names[0])
    
    def estimate_transition_matrix(
        self,
        states: List[str],
        smoothing: float = 1e-6
    ) -> np.ndarray:
        """
        Estimate transition matrix P from empirical state sequence.
        
        Builds counts C[i, j] from (state_t, state_{t+1}) pairs,
        then row-normalizes to get P.
        
        Args:
            states: Time-ordered list of state names
            smoothing: Small value to add to all transitions (prevents zero rows)
        
        Returns:
            Transition matrix P (n_states × n_states), row-stochastic
        """
        # Convert states to indices
        state_indices = [self.encode_state(s) for s in states]
        
        # Initialize count matrix
        C = np.zeros((self.n_states, self.n_states), dtype=float)
        
        # Count transitions
        for t in range(len(state_indices) - 1):
            i = state_indices[t]
            j = state_indices[t + 1]
            C[i, j] += 1.0
        
        # Add smoothing to avoid zero rows
        C = C + smoothing
        
        # Row-normalize to get transition probabilities
        row_sums = C.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)  # Avoid division by zero
        P = C / row_sums
        
        return P
    
    def estimate_stationary_distribution(
        self,
        P: np.ndarray,
        tol: float = 1e-10,
        max_iter: int = 10000
    ) -> np.ndarray:
        """
        Estimate stationary distribution π solving π = πP using power iteration.
        
        Uses power iteration on P^T to find the left eigenvector with eigenvalue 1.
        
        Args:
            P: Transition matrix (n_states × n_states)
            tol: Convergence tolerance
            max_iter: Maximum iterations
        
        Returns:
            Stationary distribution π (n_states,), normalized to sum to 1
        """
        # Initialize uniform distribution
        pi = np.ones(self.n_states) / self.n_states
        
        # Power iteration: π_{k+1} = π_k P
        for iteration in range(max_iter):
            pi_new = pi @ P
            
            # Check convergence
            if np.linalg.norm(pi_new - pi) < tol:
                break
            
            pi = pi_new
        
        # Normalize (should already be normalized, but ensure)
        pi = pi / pi.sum()
        
        return pi
    
    def compute_occupation_frequencies(self, states: List[str]) -> np.ndarray:
        """
        Compute empirical occupation frequencies from state sequence.
        
        Args:
            states: List of state names
        
        Returns:
            Occupation frequencies (n_states,), normalized to sum to 1
        """
        state_indices = [self.encode_state(s) for s in states]
        
        # Count occurrences
        counts = np.zeros(self.n_states, dtype=float)
        for idx in state_indices:
            counts[idx] += 1.0
        
        # Normalize
        if counts.sum() > 0:
            counts = counts / counts.sum()
        
        return counts
    
    def estimate_generator_from_P(
        self,
        P: np.ndarray,
        rate_scale: float = 1.0
    ) -> np.ndarray:
        """
        Estimate continuous-time generator Q from discrete-time transition matrix P.
        
        Simple approximation: Q = rate_scale * (P - I)
        Ensures:
        - Off-diagonal q_ij ≥ 0
        - Diagonal q_ii = -∑_{j≠i} q_ij
        
        Args:
            P: Transition matrix (n_states × n_states)
            rate_scale: Time scale factor (default: 1.0)
        
        Returns:
            Generator matrix Q (n_states × n_states)
        """
        I = np.eye(self.n_states)
        Q = rate_scale * (P - I)
        
        # Ensure proper generator structure:
        # Off-diagonals ≥ 0, diagonal = -sum of row
        for i in range(self.n_states):
            # Set diagonal to negative sum of off-diagonals
            Q[i, i] = -np.sum(Q[i, :]) + Q[i, i]
        
        return Q
    
    def evolve_distribution(
        self,
        p0: np.ndarray,
        Q: np.ndarray,
        horizon: float,
        n_steps: int = 50
    ) -> np.ndarray:
        """
        Numerically integrate Kolmogorov forward equation dp/dt = pQ.
        
        Uses Euler method: p(t + dt) = p(t) + dt * p(t) @ Q
        
        Args:
            p0: Initial distribution (n_states,)
            Q: Generator matrix (n_states × n_states)
            horizon: Time horizon to evolve
            n_steps: Number of integration steps
        
        Returns:
            Final distribution p(horizon) (n_states,)
        """
        p = p0.copy()
        dt = horizon / n_steps
        
        for _ in range(n_steps):
            # Euler step: dp/dt = pQ
            dp = p @ Q
            p = p + dt * dp
            
            # Ensure non-negative and normalized
            p = np.maximum(p, 0.0)
            p = p / (p.sum() + 1e-10)
        
        return p
    
    def compute_markov_risk_flag(
        self,
        pi: np.ndarray,
        p_short: np.ndarray,
        threshold_pin: float = 0.7,
        threshold_expressive: float = 0.3
    ) -> Tuple[MarkovRiskFlag, float]:
        """
        Compute Markov risk flag and Kelly modifier from π and short-horizon flow.
        
        Args:
            pi: Stationary distribution (n_states,)
            p_short: Short-horizon distribution from Kolmogorov flow (n_states,)
            threshold_pin: Threshold for PIN dominance (default: 0.7)
            threshold_expressive: Threshold for expressive states (default: 0.3)
        
        Returns:
            Tuple of (risk_flag, kelly_modifier)
            - risk_flag: MarkovRiskFlag enum
            - kelly_modifier: Scalar in (0, 1] that multiplies existing Kelly fraction
        """
        # Get state indices
        pin_idx = self.state_to_idx.get("PIN", 0)
        expressive_idx = self.state_to_idx.get("EXPRESSIVE", 1)
        rupture_idx = self.state_to_idx.get("RUPTURE_CANDIDATE", 2)
        
        # Extract probabilities
        pi_pin = pi[pin_idx] if pin_idx < len(pi) else 0.0
        pi_expressive = pi[expressive_idx] if expressive_idx < len(pi) else 0.0
        pi_rupture = pi[rupture_idx] if rupture_idx < len(pi) else 0.0
        
        p_short_pin = p_short[pin_idx] if pin_idx < len(p_short) else 0.0
        p_short_expressive = p_short[expressive_idx] if expressive_idx < len(p_short) else 0.0
        p_short_rupture = p_short[rupture_idx] if rupture_idx < len(p_short) else 0.0
        
        # Decision logic
        pi_expressive_total = pi_expressive + pi_rupture
        p_short_expressive_total = p_short_expressive + p_short_rupture
        
        # STABLE_PIN: π[PIN] > threshold and p_short still PIN-dominant
        if pi_pin > threshold_pin and p_short_pin > threshold_pin:
            return MarkovRiskFlag.STABLE_PIN, 0.1  # Very conservative
        
        # DRIFTING_EXPRESSIVE: π shows expressive mass and flow is leaking from PIN
        if pi_expressive_total > threshold_expressive and p_short_pin < pi_pin:
            # Probability is flowing away from PIN toward expressive
            if p_short_expressive_total > pi_expressive_total:
                return MarkovRiskFlag.DRIFTING_EXPRESSIVE, 0.6  # Moderate
        
        # RUPTURE_DRIFT: Significant rupture probability in flow
        if p_short_rupture > 0.15 or (pi_rupture > 0.1 and p_short_rupture > pi_rupture):
            return MarkovRiskFlag.RUPTURE_DRIFT, 0.8  # Allow higher Kelly but still gated
        
        # TRANSIENT_UNCERTAIN: Early or unstable regime
        # Check if empirical occupation is far from π (would need empirical data)
        # For now, if we're not clearly in any category, be conservative
        return MarkovRiskFlag.TRANSIENT_UNCERTAIN, 0.3
    
    def analyze_state_sequence(
        self,
        states: List[str],
        current_regime: str,
        current_gate: Optional[str] = None,
        rate_scale: float = 1.0,
        horizon: float = 5.0
    ) -> Dict:
        """
        Complete Markov analysis of a state sequence.
        
        Args:
            states: Historical state sequence
            current_regime: Current regime state
            current_gate: Current gate state (optional)
            rate_scale: Time scale for generator (default: 1.0)
            horizon: Short-horizon time for Kolmogorov flow (default: 5.0)
        
        Returns:
            Dictionary with:
            - P: Transition matrix
            - pi: Stationary distribution
            - Q: Generator matrix
            - p0: Initial distribution (one-hot at current state)
            - p_short: Short-horizon distribution
            - occupation: Empirical occupation frequencies
            - risk_flag: MarkovRiskFlag
            - kelly_modifier: Kelly fraction modifier
        """
        # Estimate transition matrix
        P = self.estimate_transition_matrix(states)
        
        # Compute stationary distribution
        pi = self.estimate_stationary_distribution(P)
        
        # Estimate generator
        Q = self.estimate_generator_from_P(P, rate_scale=rate_scale)
        
        # Initial distribution: one-hot at current state
        current_idx = self.encode_state(current_regime, current_gate)
        p0 = np.zeros(self.n_states)
        p0[current_idx] = 1.0
        
        # Evolve distribution
        p_short = self.evolve_distribution(p0, Q, horizon)
        
        # Empirical occupation
        occupation = self.compute_occupation_frequencies(states)
        
        # Risk flag and Kelly modifier
        risk_flag, kelly_modifier = self.compute_markov_risk_flag(pi, p_short)
        
        return {
            'P': P,
            'pi': pi,
            'Q': Q,
            'p0': p0,
            'p_short': p_short,
            'occupation': occupation,
            'risk_flag': risk_flag,
            'kelly_modifier': kelly_modifier,
            'state_names': self.state_names,
            'state_to_idx': self.state_to_idx.copy()
        }


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute KL divergence D_KL(p || q).
    
    Args:
        p: First distribution
        q: Second distribution
    
    Returns:
        KL divergence (bits)
    """
    # Avoid zeros
    p = np.maximum(p, 1e-10)
    q = np.maximum(q, 1e-10)
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    return np.sum(p * np.log2(p / q))


def compute_mixing_time(P: np.ndarray, tol: float = 0.01) -> int:
    """
    Estimate mixing time: smallest n such that ||P^n - π|| < tol.
    
    Args:
        P: Transition matrix
        tol: Convergence tolerance
    
    Returns:
        Estimated mixing time (number of steps)
    """
    n_states = P.shape[0]
    pi = np.ones(n_states) / n_states
    
    # Power iteration to find π
    for _ in range(1000):
        pi_new = pi @ P
        if np.linalg.norm(pi_new - pi) < 1e-10:
            break
        pi = pi_new
    
    # Compute P^n and check distance to π
    P_n = np.eye(n_states)
    for n in range(1, 1000):
        P_n = P_n @ P
        # Check max row distance to π
        max_dist = np.max(np.abs(P_n - pi.reshape(1, -1)))
        if max_dist < tol:
            return n
    
    return 1000  # Did not converge within limit
