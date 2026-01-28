#!/usr/bin/env python3
"""
Generate a single dataset by name: KITTI, SO(3), Sphere, Protein, CS, or Disk
"""

from typing import Dict, Any, Literal, Optional, Tuple, List

from kitti_data import generate_kitti_dataset
from so3_data import generate_so3_dataset
from sphere_data import create_sphere_with_vector_field
from protein_data import generate_protein_dataset
from cs_data import generate_cs_dataset
from ODEin2Ddisk_data import create_training_data as create_disk_training_data, DiskODEParams


DatasetName = Literal["kitti", "so3", "sphere", "protein", "cs", "disk"]


def generate_dataset(
    dataset_name: DatasetName,
    num_samples: int = 5000,
    train_ratio: float = 0.8,  # reserved (most generators do 80/20 internally)
    kitti_data_path: Optional[str] = None,
    protein_data_path: str = "./../Casp8",
    seed: Optional[int] = None,
    # Disk ODE options (only used if dataset_name == "disk")
    disk_dt: float = 0.01,
    disk_num_steps: int = 100,      # matches old T=1.0 with dt=0.01
    disk_noise_level: float = 0.0,
) -> Dict[str, Any]:
    """
    Generate exactly one dataset specified by `dataset_name`.

    Returns
    -------
    dict
        {dataset_name: dataset_object}

        Where dataset_object is:
        - for kitti/so3/sphere/cs/disk:
            (train_init, train_final, test_init, test_final)
        - for protein:
            List[Dict]  # raw protein records
    """
    name = dataset_name.lower().strip()

    if name == "kitti":
        train_init, train_final, test_init, test_final = generate_kitti_dataset(
            num_samples=num_samples,
            data_path=kitti_data_path,
            seed=seed,
        )
        return {"kitti": (train_init, train_final, test_init, test_final)}

    if name == "so3":
        train_init, train_final, test_init, test_final = generate_so3_dataset(
            num_samples=num_samples
        )
        return {"so3": (train_init, train_final, test_init, test_final)}

    if name == "sphere":
        train_init, train_final, test_init, test_final = create_sphere_with_vector_field(
            show_plot=False,
            num_samples=num_samples,
        )
        return {"sphere": (train_init, train_final, test_init, test_final)}

    if name == "protein":
        protein_data = generate_protein_dataset(protein_data_path, num_samples = num_samples)
        return {"protein": protein_data}

    if name == "cs":
        train_init, train_final, test_init, test_final = generate_cs_dataset(
            num_samples=num_samples,
            noise_level=0.00,  # keep your default unless you expose it as an arg
        )
        return {"cs": (train_init, train_final, test_init, test_final)}

    if name == "disk":
        disk_params = DiskODEParams(
            alpha=0.5,
            rotation_strength=1.0,
            radius=1.0,
        )
        train_init, train_final, test_init, test_final = create_disk_training_data(
            num_samples=num_samples,
            num_steps=disk_num_steps,
            dt=disk_dt,
            noise_level=disk_noise_level,
            seed=seed,
            params=disk_params,
            # Closest to old behavior:
            renormalize_after_noise=False,
            project_after_step=False,
        )
        return {"disk": (train_init, train_final, test_init, test_final)}

    raise ValueError(
        f"Unknown dataset_name={dataset_name!r}. "
        "Expected one of: 'kitti', 'so3', 'sphere', 'protein', 'cs', 'disk'."
    )
