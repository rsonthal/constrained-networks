import numpy as np
from typing import Dict, Any, List, Tuple
from tqdm.auto import tqdm

eps = 1e-8

def torsion_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Compute the signed torsion (dihedral) angle defined by four 3D points.

    Geometry
    --------
    Given points p1, p2, p3, p4 ∈ R^3, form bond vectors
        b1 = p2 - p1,  b2 = p3 - p2,  b3 = p4 - p3
    and plane normals
        n1 = b1 × b2,  n2 = b2 × b3.
    The signed dihedral ϕ ∈ (-π, π] between the two planes is
        ϕ = atan2( ||b2|| * (b1 · (b2 × b3)),  (n1 · n2) ).
    This is the standard “atan2” dihedral used in structural biology.

    Returns
    -------
    float
        The torsion angle in radians, in (-π, π].

    Notes
    -----
    - Matches the conventional definitions for protein backbone torsions
      (e.g., φ, ψ, ω) when (p1, p2, p3, p4) are the appropriate atoms.
    - See AlphaFold SI §1.9.1 for angle representation on S¹ and JCBN/IUPAC
      §3.2.1 for which atom quadruples define φ/ψ/ω.

    Examples
    --------
    >>> a = np.array([0.0, 0.0, 0.0])
    >>> b = np.array([1.0, 0.0, 0.0])
    >>> c = np.array([1.0, 1.0, 0.0])
    >>> d = np.array([1.0, 1.0, 1.0])
    >>> angle = torsion_angle(a, b, c, d)  # radians
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    y = np.linalg.norm(b2) * np.dot(b1, np.cross(b2, b3))
    x = np.dot(n1, n2)

    return float(np.arctan2(y, x))

def backbone_frame_from_atoms(xN: np.ndarray, xCA: np.ndarray, xC: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the per-residue backbone rigid transform T = (R, t) from the three
    backbone atoms (N, Cα, C), following AlphaFold Supplementary Information
    §1.8.1, Algorithm 21 (“Construction of frames from ground-truth atom positions”).

    Construction (Gram–Schmidt on the N–Cα–C triad)
    -----------------------------------------------
    Let v1 = xC  - xCA
        v2 = xN  - xCA
    e1 = v1 / ||v1||                                  # axis roughly toward C
    e2 = (v2 - (e1·v2) e1) / ||v2 - (e1·v2) e1||      # in the N–Cα–C plane, ⟂ e1
    e3 = e1 × e2                                      # right-hand completion
    R  = [e1, e2, e3]                                 # columns form an SO(3) basis
    t  = xCA                                          # frame origin at Cα

    Returns
    -------
    R : (3, 3) ndarray
        Rotation matrix with det(R) = +1 (if det < 0, flip e3 as per Alg. 21).
    t : (3,) ndarray
        Translation vector (the Cα position).

    Notes
    -----
    - This is the exact backbone “local frame” used throughout AlphaFold’s
      structure module and FAPE (see §1.8.1 and §1.9.2).
    - The transform maps local → global as x_global = R @ x_local + t.

    Example
    -------
    >>> N  = np.array([ 1.2, -0.8,  0.3])
    >>> CA = np.array([ 0.0,  0.0,  0.0])
    >>> C  = np.array([ 1.5,  0.7, -0.2])
    >>> R, t = backbone_frame_from_atoms(N, CA, C)
    """
    v1 = xC - xCA
    v2 = xN - xCA

    n1 = np.linalg.norm(v1)
    if n1 < eps:
        return np.full((3,3), np.nan), np.full((3,), np.nan)

    e1 = v1 / n1

    v2_orth = v2 - np.dot(e1, v2) * e1
    n2 = np.linalg.norm(v2_orth)
    if n2 < eps:
        return np.full((3,3), np.nan), np.full((3,), np.nan)

    e2 = v2_orth / n2
    e3 = np.cross(e1, e2)
    e3 = e3 / (np.linalg.norm(e3) + eps)

    R = np.column_stack((e1, e2, e3))
    if np.linalg.det(R) < 0.0:
        R[:, 2] = -R[:, 2]

    return R, xCA

def se3_from_R_t(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Build a 4x4 homogeneous SE(3) matrix from (R, t).
    R: (3,3) rotation in SO(3)
    t: (3,)  translation
    Returns T: (4,4) with [[R, t],[0,0,0,1]]
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T

def apply_se3(T: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Apply SE(3) transform to 3D points.
    x: (..., 3) array of points
    Returns: (..., 3) transformed points
    """
    return (x @ T[:3, :3].T) + T[:3, 3]

def compose_se3(T2: np.ndarray, T1: np.ndarray) -> np.ndarray:
    """
    Composition T = T2 ∘ T1 (apply T1 then T2), via 4x4 matrix product.
    """
    return T2 @ T1

def inverse_se3(T: np.ndarray) -> np.ndarray:
    """
    Group inverse: T^{-1} = [[R^T, -R^T t],[0,0,0,1]].
    """
    R = T[:3, :3]
    t = T[:3,  3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3,  3] = -R.T @ t
    return Tinv

from typing import List, Dict, Any, Tuple

def finalize_current(current: Dict[str, Any]) -> Dict[str, Any]:
    assert len(current["tertiary_rows"]) == 3
    assert len(current["tertiary_rows"][0]) == len(current["tertiary_rows"][1])
    assert len(current["tertiary_rows"][0]) == len(current["tertiary_rows"][2])

    n_res = len(current["mask"])
    assert 3 * n_res == len(current["tertiary_rows"][0])

    xyz = np.asarray(current["tertiary_rows"], dtype=np.float32)  # (3, 3*n_res)
    xyz_res_atom_coord = xyz.reshape(3, n_res, 3).transpose(1, 2, 0)  # (n_res, 3, 3)

    current["n_res"] = n_res
    current["xyz_res_atom_coord"] = xyz_res_atom_coord
    current["N_xyz"]  = xyz_res_atom_coord[:, 0, :]
    current["CA_xyz"] = xyz_res_atom_coord[:, 1, :]
    current["C_xyz"]  = xyz_res_atom_coord[:, 2, :]
    del current["tertiary_rows"]
    return current


def parse_casp8_file_with_mask(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a CASP/ProteinNet text file and collect minimal fields needed downstream.
    Returns a list of dicts with:
        - 'id'          : str
        - 'coordinates' : List[float]  (TERTIARY as a flat list; will reshape later)
        - 'mask'        : str of '+'/'-' characters (one per residue)
    """
    proteins: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    section: str | None = None

    with open(file_path, "r") as f:
        total_lines = sum(1 for _ in f)
                          
    with open(file_path, "r") as f:
        it = tqdm(f, total=total_lines, desc="Parsing ProteinNet", unit="line")
        for raw in it:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("[ID]"):
                if current is not None:
                    proteins.append(finalize_current(current))
                    if len(proteins) % 1000 == 0:
                        it.set_postfix(records=len(proteins))
                current = {"id": None, "mask": "", "tertiary_rows": []}
                section = "id"
                continue
            elif line.startswith("[PRIMARY]"):
                section = "primary";  continue
            elif line.startswith("[EVOLUTIONARY]"):
                section = "evolutionary";  continue
            elif line.startswith("[TERTIARY]"):
                section = "tertiary";  continue
            elif line.startswith("[MASK]"):
                section = "mask";  continue

            if current is None:
                continue

            if section == "id":
                current["id"] = line
            elif section == "tertiary":
                current["tertiary_rows"].append(list(map(float, line.split())))
            elif section == "mask":
                # ProteinNet uses a single line of '+'/'-' for residues
                current["mask"] = line

    if current is not None:
        proteins.append(finalize_current(current))
    print(f"Parsed {len(proteins)} records")
    return proteins


def records_to_frames_angles(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert parsed records into per-residue frames (R,t) and backbone torsions (φ,ψ,ω).

    Geometry & references:
      - Frames T_i=(R_i,t_i) from (N, Cα, C) via Gram–Schmidt (AlphaFold SI §1.8.1, Alg. 21).
      - Torsions (IUPAC/JCBN §3.2.1):
          φ_i  = dihedral(C_{i-1}, N_i,   Cα_i,  C_i)
          ψ_i  = dihedral(N_i,    Cα_i,  C_i,   N_{i+1})
          ω_i  = dihedral(Cα_i,   C_i,   N_{i+1}, Cα_{i+1})
      - Ends: φ_0, ψ_{N-1}, ω_{N-1} undefined → NaN.
      - Mask: any torsion involving a masked residue → NaN; masked residues’ (R,t) → NaN rows.

    Returns per record:
        {
          'id'     : str,
          'R'      : (N, 3, 3) rotation matrices (NaN where masked),
          't'      : (N, 3) translations (Cα positions; NaN where masked),
          'angles' : (N, 3) rows = [φ, ψ, ω] in radians (NaN where undefined),
          'mask'   : (N,) bool
        }
    """
    out: List[Dict[str, Any]] = []
    it = tqdm(records, desc="Frames+angles", unit="protein") 

    for rec in it:
        N_res = rec["n_res"]

        N_xyz  = rec["N_xyz"]   # (N_res, 3)
        CA_xyz = rec["CA_xyz"]  # (N_res, 3)
        C_xyz  = rec["C_xyz"]   # (N_res, 3)
        
        mask_str = rec["mask"].strip()
        mask = np.fromiter((c == "+" for c in mask_str), dtype=bool, count=N_res)

        # --- Frames (Alg. 21) ---
        R_all = np.full((N_res, 3, 3), np.nan, dtype=float)
        t_all = np.full((N_res, 3), np.nan, dtype=float)
        for i in range(N_res):
            if mask[i]:
                R_i, t_i = backbone_frame_from_atoms(N_xyz[i], CA_xyz[i], C_xyz[i])
                R_all[i], t_all[i] = R_i, t_i

        # --- Torsions (φ, ψ, ω) ---
        phi   = np.full(N_res, np.nan, dtype=float)
        psi   = np.full(N_res, np.nan, dtype=float)
        omega = np.full(N_res, np.nan, dtype=float)

        # φ_i uses residues (i-1, i)
        for i in range(1, N_res):
            if mask[i-1] and mask[i]:
                phi[i] = torsion_angle(C_xyz[i-1], N_xyz[i], CA_xyz[i], C_xyz[i])

        # ψ_i, ω_i use residues (i, i+1)
        for i in range(N_res - 1):
            if mask[i] and mask[i+1]:
                psi[i]   = torsion_angle(N_xyz[i], CA_xyz[i], C_xyz[i], N_xyz[i+1])
                omega[i] = torsion_angle(CA_xyz[i], C_xyz[i], N_xyz[i+1], CA_xyz[i+1])

        angles = np.stack([phi, psi, omega], axis=1)  # (N, 3)

        out.append(
            {
                "id": rec["id"],
                "R": R_all,
                "t": t_all,
                "angles": angles,
                "mask": mask,
            }
        )
        
    print(f"Processed {len(out)} records into frames/angles")
    return out

def frame_ok(p):
    mask = p["mask"]
    R_ok = np.isfinite(p["R"]).all(axis=(1, 2))
    t_ok = np.isfinite(p["t"]).all(axis=1)
    return mask & R_ok & t_ok

def generate_protein_dataset(file_path: str, num_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a protein dataset by loading protein records, converting to SE(3) matrices,
    and creating temporal prediction pairs.
    
    This function follows the same pattern as `generate_kitti_dataset()` and 
    `generate_so3_dataset()` - it's the main entry point for generating
    protein data for model training.
    
    Parameters
    ----------
    file_path : str
        Path to ProteinNet/CASP text file.
    num_samples : int, default=5000
        Number of (input, target) pairs to generate. If the available data has
        fewer pairs, all available pairs will be used.
        
    Returns
    -------
    train_init, train_final, test_init, test_final : np.ndarray
        Arrays of shape (N_train, 4, 4) / (N_test, 4, 4) with SE(3) matrices.
        train_init/test_init are input matrices, train_final/test_final are targets.
        Uses an 80/20 train/test split.
    """

    rng = np.random.default_rng(0)
    
    # Load and process protein records
    processed = records_to_frames_angles(parse_casp8_file_with_mask(file_path))
    
    # Convert to SE(3) matrices and create consecutive residue pairs
    
    # count
    total_pairs = sum(int(np.sum(frame_ok(p)[:-1] & frame_ok(p)[1:])) for p in processed)
    print(f"Total valid consecutive pairs available: {total_pairs}")
    
    input_matrices  = np.zeros((total_pairs, 4, 4), dtype=np.float32)
    target_matrices = np.zeros((total_pairs, 4, 4), dtype=np.float32)
    input_matrices[:, 3, 3]  = 1.0
    target_matrices[:, 3, 3] = 1.0
    
    k = 0
    it = tqdm(processed, desc="Building SE(3) pairs", unit="protein", mininterval=0.5)

    for p in processed:
        R, t, mask = p["R"], p["t"], p["mask"]
        ok = frame_ok(p)
        idx = np.where(ok[:-1] & ok[1:])[0]
        
        m = len(idx)
        if m == 0:
            continue
    
        input_matrices[k:k+m, :3, :3] = R[idx].astype(np.float32, copy=False)
        input_matrices[k:k+m, :3,  3] = t[idx].astype(np.float32, copy=False)
    
        target_matrices[k:k+m, :3, :3] = R[idx+1].astype(np.float32, copy=False)
        target_matrices[k:k+m, :3,  3] = t[idx+1].astype(np.float32, copy=False)
    
        k += m
        it.set_postfix(pairs=k)
    
    assert k == total_pairs
    
    N = input_matrices.shape[0]
    
    if N > num_samples:
        idx = rng.choice(N, size=num_samples, replace=False)
        input_matrices, target_matrices = input_matrices[idx], target_matrices[idx]
        N = num_samples
    else:
        num_samples = N  # optional, only if you want requested == actual
    
    perm = rng.permutation(N)
    input_matrices, target_matrices = input_matrices[perm], target_matrices[perm]
    
    Ntr = int(0.8 * N)
    train_init, train_final = input_matrices[:Ntr], target_matrices[:Ntr]
    test_init,  test_final  = input_matrices[Ntr:], target_matrices[Ntr:]
    
    print(f"Generated {N} protein SE(3) pairs.")
    print(f"Train: {train_init.shape[0]}   Test: {test_init.shape[0]}")
    
        
    return train_init, train_final, test_init, test_final

