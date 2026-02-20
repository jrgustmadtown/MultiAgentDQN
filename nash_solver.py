"""
Nash Equilibrium Solver for Zero-Sum Games

Implements minimax computation for 2-player zero-sum games.
For car game: Q(s, a_A, a_B) where Car A maximizes, Car B minimizes.

Optimized for speed with vectorized numpy operations.
"""

import numpy as np


def compute_minimax_value(q_matrix):
    """
    Compute the minimax value for a zero-sum game (vectorized).
    
    Returns V* = max_a min_b Q(s, a, b), handling -inf for invalid actions.
    """
    # Replace -inf with large positive for min (so they won't be chosen)
    q_for_min = np.where(np.isfinite(q_matrix), q_matrix, np.inf)
    worst_case = np.min(q_for_min, axis=1)  # [4]
    
    # If all B actions were invalid for an A action, worst_case[a] = inf
    # Replace inf with -inf so A won't pick that action
    worst_case = np.where(worst_case == np.inf, -np.inf, worst_case)
    
    # A picks max, ignoring -inf
    valid = np.isfinite(worst_case)
    if np.any(valid):
        return np.max(worst_case[valid])
    return -np.inf


def compute_nash_equilibrium(q_matrix):
    """
    Compute Nash equilibrium for a zero-sum game (vectorized).
    
    Returns (nash_value, best_action_A, best_action_B).
    """
    # Replace -inf with +inf for min operation (B won't pick invalid actions)
    q_for_min = np.where(np.isfinite(q_matrix), q_matrix, np.inf)
    worst_case = np.min(q_for_min, axis=1)  # [4]
    
    # If all B actions invalid for row, set to -inf so A won't pick
    worst_case = np.where(worst_case == np.inf, -np.inf, worst_case)
    
    # A picks the action with max worst-case
    valid_A = np.isfinite(worst_case)
    if np.any(valid_A):
        best_action_A = np.argmax(np.where(valid_A, worst_case, -np.inf))
        nash_value = worst_case[best_action_A]
    else:
        best_action_A = 0
        nash_value = -np.inf
    
    # B's best response: minimize in A's chosen row, among valid actions
    row = q_matrix[best_action_A, :]
    row_for_min = np.where(np.isfinite(row), row, np.inf)
    best_action_B = np.argmin(row_for_min)
    
    return nash_value, best_action_A, best_action_B


def compute_maximin_value(q_matrix):
    """
    Compute the maximin value (alternative perspective for zero-sum games).
    
    For Player B (minimizer):
    - For each action of B, find best case (max over A's actions)
    - B picks the action that minimizes the best case
    
    In zero-sum games, minimax value = maximin value (von Neumann theorem).
    
    Args:
        q_matrix: [5, 5] numpy array
    
    Returns:
        maximin_value: Should equal minimax value
    """
    best_case_for_each_b = np.max(q_matrix, axis=0)  # Max over A's actions
    maximin_value = np.min(best_case_for_each_b)
    return maximin_value


# Test functions
if __name__ == "__main__":
    print("Testing Nash Solver\n")
    
    # Test 1: Simple game where A should go UP, B should go DOWN
    q1 = np.array([
        [10,  5,  3,  2,  1],  # UP
        [ 8,  4,  2,  1,  0],  # DOWN
        [ 6,  3,  1,  0, -1],  # LEFT
        [ 4,  2,  0, -1, -2],  # RIGHT
        [ 2,  1, -1, -2, -3],  # STAY
    ])
    
    print("Test 1: Simple game")
    print("Q-matrix:")
    print(q1)
    nash_val, action_A, action_B = compute_nash_equilibrium(q1)
    print(f"\nNash value: {nash_val}")
    print(f"Best action A: {action_A} (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY)")
    print(f"Best action B: {action_B}")
    print(f"Maximin value: {compute_maximin_value(q1)} (should equal Nash value)")
    
    # Test 2: Saddle point game
    q2 = np.array([
        [3, 2, 4, 1, 2],
        [5, 4, 6, 3, 4],
        [2, 1, 3, 0, 1],
        [4, 3, 5, 2, 3],
        [1, 0, 2, -1, 0],
    ])
    
    print("\n\nTest 2: Saddle point game")
    print("Q-matrix:")
    print(q2)
    nash_val, action_A, action_B = compute_nash_equilibrium(q2)
    print(f"\nNash value: {nash_val}")
    print(f"Best action A: {action_A}")
    print(f"Best action B: {action_B}")
    print(f"Maximin value: {compute_maximin_value(q2)}")
    
    # Test 3: Uniform game (all same value - any action works)
    q3 = np.ones((5, 5)) * 5.0
    
    print("\n\nTest 3: Uniform game")
    print("Q-matrix:")
    print(q3)
    nash_val, action_A, action_B = compute_nash_equilibrium(q3)
    print(f"\nNash value: {nash_val}")
    print(f"Best action A: {action_A}")
    print(f"Best action B: {action_B}")
