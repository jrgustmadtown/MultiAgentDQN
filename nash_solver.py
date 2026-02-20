"""
Nash Equilibrium Solver for Zero-Sum Games

Implements minimax computation for 2-player zero-sum games.
For car game: Q(s, a_A, a_B) where Car A maximizes, Car B minimizes.
"""

import numpy as np


def compute_minimax_value(q_matrix):
    """
    Compute the minimax value for a zero-sum game.
    
    For a 2-player zero-sum game:
    - Player A (row player) maximizes
    - Player B (column player) minimizes
    - Car game: Car A tries to maximize Q, Car B tries to minimize Q
    
    Returns the value of the game V* = max_a min_b Q(s, a, b)
    
    Args:
        q_matrix: [5, 5] numpy array where q_matrix[a_A, a_B] = Q(s, a_A, a_B)
    
    Returns:
        nash_value: The minimax value of the game
    """
    # For each action of A, find worst case (min over B's actions)
    worst_case_for_each_a = np.min(q_matrix, axis=1)  # [5] array
    
    # A picks the action that maximizes the worst case
    nash_value = np.max(worst_case_for_each_a)
    
    return nash_value


def compute_minimax_strategy(q_matrix):
    """
    Compute the minimax strategy (which action A should play).
    
    Args:
        q_matrix: [5, 5] numpy array where q_matrix[a_A, a_B] = Q(s, a_A, a_B)
    
    Returns:
        best_action_A: The action that achieves minimax value
    """
    worst_case_for_each_a = np.min(q_matrix, axis=1)
    best_action_A = np.argmax(worst_case_for_each_a)
    return best_action_A


def compute_nash_equilibrium(q_matrix):
    """
    Compute Nash equilibrium for a zero-sum game.
    
    For simple 2-player zero-sum games, Nash equilibrium is the minimax solution.
    Returns the value and both players' best responses.
    
    Args:
        q_matrix: [5, 5] numpy array where q_matrix[a_A, a_B] = Q(s, a_A, a_B)
    
    Returns:
        nash_value: The value of the game
        best_action_A: Best action for player A
        best_action_B: Best action for player B (against A's strategy)
    """
    # A's minimax strategy
    worst_case_for_A = np.min(q_matrix, axis=1)  # Min over B's actions
    best_action_A = np.argmax(worst_case_for_A)
    nash_value = worst_case_for_A[best_action_A]
    
    # B's best response to A's minimax action
    # B wants to minimize, so pick action with lowest Q value when A plays best_action_A
    best_action_B = np.argmin(q_matrix[best_action_A, :])
    
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
