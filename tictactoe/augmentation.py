"""
Data augmentation utilities for tic-tac-toe.
Implements rotations and reflections to exploit board symmetry.
"""
import numpy as np
from config import Config as cfg

def rotate_90(board_2d):
    """Rotate board 90 degrees clockwise."""
    return np.rot90(board_2d, k=-1)

def rotate_180(board_2d):
    """Rotate board 180 degrees."""
    return np.rot90(board_2d, k=2)

def rotate_270(board_2d):
    """Rotate board 270 degrees clockwise (90 counter-clockwise)."""
    return np.rot90(board_2d, k=1)

def reflect_horizontal(board_2d):
    """Reflect board horizontally."""
    return np.fliplr(board_2d)

def reflect_vertical(board_2d):
    """Reflect board vertically."""
    return np.flipud(board_2d)

def reflect_diagonal(board_2d):
    """Reflect board across main diagonal."""
    return board_2d.T

def reflect_anti_diagonal(board_2d):
    """Reflect board across anti-diagonal."""
    return np.fliplr(np.flipud(board_2d)).T

def transform_action(action_index, transform_type):
    """
    Transform action index according to transformation.
    
    Args:
        action_index: Original action index (0-80)
        transform_type: Type of transformation ('rotate_90', 'rotate_180', etc.)
    
    Returns:
        Transformed action index
    """
    row = action_index // 9
    col = action_index % 9
    
    if transform_type == 'rotate_90':
        # (row, col) -> (col, 8-row)
        new_row, new_col = col, 8 - row
    elif transform_type == 'rotate_180':
        # (row, col) -> (8-row, 8-col)
        new_row, new_col = 8 - row, 8 - col
    elif transform_type == 'rotate_270':
        # (row, col) -> (8-col, row)
        new_row, new_col = 8 - col, row
    elif transform_type == 'reflect_horizontal':
        # (row, col) -> (row, 8-col)
        new_row, new_col = row, 8 - col
    elif transform_type == 'reflect_vertical':
        # (row, col) -> (8-row, col)
        new_row, new_col = 8 - row, col
    elif transform_type == 'reflect_diagonal':
        # (row, col) -> (col, row)
        new_row, new_col = col, row
    elif transform_type == 'reflect_anti_diagonal':
        # (row, col) -> (8-col, 8-row)
        new_row, new_col = 8 - col, 8 - row
    else:
        return action_index
    
    return new_row * 9 + new_col

def transform_policy(policy, transform_type):
    """
    Transform policy distribution according to transformation.
    
    Args:
        policy: Original policy distribution (81 values)
        transform_type: Type of transformation
    
    Returns:
        Transformed policy distribution
    """
    transformed_policy = np.zeros_like(policy)
    for i in range(len(policy)):
        if policy[i] > 0:
            new_index = transform_action(i, transform_type)
            transformed_policy[new_index] = policy[i]
    return transformed_policy

def augment_data(state_flat, policy, transform_type):
    """
    Augment a single data point with a transformation.
    
    Args:
        state_flat: Flat board state (81 values)
        policy: Policy distribution (81 values)
        transform_type: Type of transformation
    
    Returns:
        (augmented_state, augmented_policy)
    """
    # Reshape to 2D
    board_2d = state_flat.reshape(9, 9)
    
    # Apply transformation to board
    if transform_type == 'rotate_90':
        board_2d = rotate_90(board_2d)
    elif transform_type == 'rotate_180':
        board_2d = rotate_180(board_2d)
    elif transform_type == 'rotate_270':
        board_2d = rotate_270(board_2d)
    elif transform_type == 'reflect_horizontal':
        board_2d = reflect_horizontal(board_2d)
    elif transform_type == 'reflect_vertical':
        board_2d = reflect_vertical(board_2d)
    elif transform_type == 'reflect_diagonal':
        board_2d = reflect_diagonal(board_2d)
    elif transform_type == 'reflect_anti_diagonal':
        board_2d = reflect_anti_diagonal(board_2d)
    
    # Flatten back
    augmented_state = board_2d.flatten()
    
    # Transform policy
    augmented_policy = transform_policy(policy, transform_type)
    
    return augmented_state, augmented_policy

def get_augmentations():
    """Get list of all possible augmentation types."""
    return [
        'rotate_90',
        'rotate_180',
        'rotate_270',
        'reflect_horizontal',
        'reflect_vertical',
        'reflect_diagonal',
        'reflect_anti_diagonal'
    ]

