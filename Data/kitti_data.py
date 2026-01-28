import numpy as np
import os
import glob
from typing import Tuple, Optional

# Self-contained KITTI data loading and processing module
# Follows the same pattern as sphere_data.py and so3_data.py

def load_kitti_oxts_data(data_path: Optional[str] = None) -> np.ndarray:
    """
    Load KITTI OXTS data from text files.
    
    Args:
        data_path: Path to directory containing OXTS .txt files
        
    Returns:
        Array of shape (num_frames, 30) with OXTS data
    """
    if data_path is None:
        # Try multiple possible paths
        possible_paths = [
            os.path.expanduser("~/Downloads/Kitti/oxts/data"),
            "/content/drive/MyDrive/Kitti/oxts/data",
            "/root/Downloads/Kitti/oxts/data",
            "./Kitti/oxts/data",
            "Kitti/oxts/data",
            "KITTI Data/Kitti/oxts/data",
            "../KITTI Data/Kitti/oxts/data",
            "../Projected Transformer/KITTI Data/Kitti/oxts/data"
        ]
        
        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
                
        if data_path is None:
            error_msg = (
                "KITTI OXTS data not found. Please provide the path to your KITTI OXTS data directory.\n"
                "Expected format: Directory containing .txt files with OXTS data.\n"
                "Searched locations:\n"
            )
            for path in possible_paths:
                error_msg += f"  - {path}\n"
            error_msg += (
                "\nTo fix this:\n"
                "  1. Download KITTI dataset from http://www.cvlibs.net/datasets/kitti/\n"
                "  2. Extract the OXTS data files to a directory\n"
                "  3. Pass the path via data_path parameter or place data in one of the searched locations"
            )
            raise FileNotFoundError(error_msg)
    
    # Get all .txt files sorted by name
    files = sorted(glob.glob(os.path.join(data_path, "*.txt")))
    
    if len(files) < 11:
        raise ValueError(
            f"Insufficient KITTI OXTS data files found. Need at least 11 files for temporal prediction, "
            f"but found {len(files)} files in {data_path}.\n"
            "Please ensure you have a complete KITTI OXTS dataset."
        )
    
    print(f"Found {len(files)} OXTS data files in {data_path}")
    
    # Load all data
    oxts_data = []
    for file_path in files:
        data = np.loadtxt(file_path)
        oxts_data.append(data)
    
    oxts_data = np.array(oxts_data)
    print(f"Loaded OXTS data with shape: {oxts_data.shape}")
    
    return oxts_data

def oxts_to_se3(oxts_data: np.ndarray) -> np.ndarray:
    """
    Convert OXTS data to SE(3) matrices.
    
    OXTS format: [lat, lon, alt, roll, pitch, yaw, vn, ve, vf, vl, vu, 
                  ax, ay, az, af, al, au, wx, wy, wz, wf, wl, wu, 
                  pos_accuracy, vel_accuracy, navstat, numsats, posmode, velmode, orimode]
    
    Args:
        oxts_data: Array of shape (num_frames, 30)
        
    Returns:
        Array of shape (num_frames, 4, 4) with SE(3) matrices
    """
    num_frames = len(oxts_data)
    se3_matrices = np.zeros((num_frames, 4, 4), dtype=np.float32)
    
    for i, oxts in enumerate(oxts_data):
        # Extract roll, pitch, yaw (indices 3, 4, 5)
        roll, pitch, yaw = oxts[3], oxts[4], oxts[5]
        
        # Create rotation matrix from Euler angles (ZYX convention)
        cos_roll, sin_roll = np.cos(roll), np.sin(roll)
        cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        
        # Rotation matrix (ZYX Euler angles)
        R = np.array([
            [cos_yaw * cos_pitch, cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll, cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll],
            [sin_yaw * cos_pitch, sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll, sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll],
            [-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll]
        ], dtype=np.float32)
        
        # Use velocity components (vn, ve, vf) as translation
        translation = np.array([oxts[6], oxts[7], oxts[8]], dtype=np.float32)  # vn, ve, vf
        
        # Construct SE(3) matrix
        se3_matrix = np.eye(4, dtype=np.float32)
        se3_matrix[:3, :3] = R
        se3_matrix[:3, 3] = translation
        
        se3_matrices[i] = se3_matrix
    
    return se3_matrices

def create_temporal_pairs(se3_matrices: np.ndarray, offset: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create temporal prediction pairs from SE(3) matrices.
    
    Args:
        se3_matrices: Array of shape (num_frames, 4, 4)
        offset: Temporal offset for prediction (frame n -> frame n+offset)
        
    Returns:
        Tuple of (input_matrices, target_matrices) both of shape (num_pairs, 4, 4)
    """
    num_matrices = len(se3_matrices)
    max_valid_n = num_matrices - offset
    
    if max_valid_n < 1:
        raise ValueError(f"Not enough matrices for offset {offset}. Need at least {offset + 1} matrices, got {num_matrices}")
    
    # Create pairs: (SE3_input, SE3_target)
    input_matrices = []
    target_matrices = []
    
    for n in range(max_valid_n):
        input_matrices.append(se3_matrices[n])
        target_matrices.append(se3_matrices[n + offset])
    
    input_matrices = np.array(input_matrices)
    target_matrices = np.array(target_matrices)
    
    return input_matrices, target_matrices

def validate_se3_matrices(se3_matrices: np.ndarray) -> dict:
    """
    Validate SE(3) matrices and compute constraint satisfaction metrics.
    
    Args:
        se3_matrices: Array of shape (num_matrices, 4, 4)
        
    Returns:
        Dictionary with validation metrics
    """
    num_matrices = se3_matrices.shape[0]
    
    # Extract rotation matrices and translations
    R_matrices = se3_matrices[:, :3, :3]
    t_vectors = se3_matrices[:, :3, 3]
    bottom_rows = se3_matrices[:, 3, :]
    
    # Check rotation matrix properties
    determinants = np.linalg.det(R_matrices)
    is_orthogonal = np.allclose(R_matrices @ R_matrices.transpose(0, 2, 1), 
                               np.eye(3)[None, :, :], atol=1e-3)
    
    # Check bottom row
    correct_bottom_row = np.allclose(bottom_rows, 
                                   np.array([0., 0., 0., 1.])[None, :], atol=1e-3)
    
    # Compute metrics
    avg_determinant = np.mean(determinants)
    std_determinant = np.std(determinants)
    det_range = [np.min(determinants), np.max(determinants)]
    
    # Orthogonality error
    orthogonality_errors = []
    for i in range(num_matrices):
        error = np.linalg.norm(R_matrices[i] @ R_matrices[i].T - np.eye(3))
        orthogonality_errors.append(error)
    
    avg_orthogonality_error = np.mean(orthogonality_errors)
    
    return {
        'num_matrices': num_matrices,
        'avg_determinant': avg_determinant,
        'std_determinant': std_determinant,
        'determinant_range': det_range,
        'is_orthogonal': is_orthogonal,
        'correct_bottom_row': correct_bottom_row,
        'avg_orthogonality_error': avg_orthogonality_error
    }

def generate_kitti_dataset(num_samples: int = 5000, offset: int = 10, 
                          data_path: Optional[str] = None, 
                          seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a KITTI dataset by loading OXTS data, converting to SE(3) matrices,
    and creating temporal prediction pairs.
    
    This function follows the same pattern as `generate_so3_dataset()` and 
    `create_sphere_with_vector_field()` - it's the main entry point for generating
    KITTI data for model training.
    
    Parameters
    ----------
    num_samples : int, default=5000
        Number of (input, target) pairs to generate. If the available data has
        fewer pairs, all available pairs will be used.
    offset : int, default=10
        Temporal offset for prediction (frame n -> frame n+offset).
    data_path : str or None, default=None
        Path to KITTI OXTS data directory. If None, searches common locations.
        Raises FileNotFoundError if data is not found.
    seed : int or None, default=None
        Random seed for reproducibility. If None, uses current random state.
        
    Returns
    -------
    train_init, train_final, test_init, test_final : np.ndarray
        Arrays of shape (N_train, 4, 4) / (N_test, 4, 4) with SE(3) matrices.
        train_init/test_init are input matrices, train_final/test_final are targets.
        Uses an 80/20 train/test split.
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"Generating KITTI dataset with {num_samples} samples "
          f"(offset={offset}, data_path={data_path}) ...")
    
    # Load OXTS data
    oxts_data = load_kitti_oxts_data(data_path)
    
    # Convert to SE(3) matrices
    se3_matrices = oxts_to_se3(oxts_data)
    
    # Create temporal pairs
    input_matrices, target_matrices = create_temporal_pairs(se3_matrices, offset=offset)
    
    # If we have more pairs than needed, randomly sample
    if len(input_matrices) > num_samples:
        indices = np.random.choice(len(input_matrices), num_samples, replace=False)
        input_matrices = input_matrices[indices]
        target_matrices = target_matrices[indices]
    elif len(input_matrices) < num_samples:
        print(f"Warning: Only {len(input_matrices)} pairs available, requested {num_samples}")
        print(f"Using all {len(input_matrices)} available pairs")
        num_samples = len(input_matrices)
    
    # 80/20 split (deterministic)
    Ntr = int(0.8 * num_samples)
    train_init, test_init = input_matrices[:Ntr], input_matrices[Ntr:]
    train_final, test_final = target_matrices[:Ntr], target_matrices[Ntr:]
    
    print(f"Generated {num_samples} KITTI SE(3) pairs.")
    print(f"Train: {len(train_init)}   Test: {len(test_init)}")
    
    # Quick diagnostics
    print("\nInput SE(3) matrices validation:")
    input_validation = validate_se3_matrices(train_init)
    print(f"  Average determinant: {input_validation['avg_determinant']:.6f} ± {input_validation['std_determinant']:.6f}")
    print(f"  Are orthogonal: {input_validation['is_orthogonal']}")
    print(f"  Average orthogonality error: {input_validation['avg_orthogonality_error']:.6f}")
    
    print("\nTarget SE(3) matrices validation:")
    target_validation = validate_se3_matrices(train_final)
    print(f"  Average determinant: {target_validation['avg_determinant']:.6f} ± {target_validation['std_determinant']:.6f}")
    print(f"  Are orthogonal: {target_validation['is_orthogonal']}")
    print(f"  Average orthogonality error: {target_validation['avg_orthogonality_error']:.6f}")
    
    return train_init, train_final, test_init, test_final

