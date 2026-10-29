"""
NEXUS-OPS | QFF 2026 Phase 2
Central Configuration & Resource Resolver
Provides robust, environment-aware path discovery for the project root,
official Challenge Kit, datasets, noise model, and benchmark tools.
"""
import os
from pathlib import Path


def get_project_root() -> Path:
    """Returns the absolute root directory of NEXUS_OPS_PS_Q1."""
    return Path(__file__).resolve().parents[2]


def get_challenge_kit_path() -> Path:
    """
    Locates the official QFF2026_QML_Challenge_Kit with prioritized search order:
    1. Environment variable QFF_CHALLENGE_KIT (if set and valid)
    2. Sibling directory: ../QFF2026_QML_Track/QFF2026_QML_Challenge_Kit
    3. Direct parent sibling: ../QFF2026_QML_Challenge_Kit
    4. Internal embedded kit: ./QFF2026_QML_Challenge_Kit
    """
    env_path = os.environ.get("QFF_CHALLENGE_KIT")
    if env_path:
        p = Path(env_path).resolve()
        if p.exists() and (p / "schemas" / "processor_d_schema.json").exists():
            return p

    root = get_project_root()
    candidates = [
        root.parent / "QFF2026_QML_Track" / "QFF2026_QML_Challenge_Kit",
        root.parent / "QFF2026_QML_Challenge_Kit",
        root / "QFF2026_QML_Challenge_Kit"
    ]

    for cand in candidates:
        if cand.exists() and (cand / "schemas" / "processor_d_schema.json").exists():
            return cand.resolve()

    searched_str = "\n".join(f"  - {str(c)}" for c in candidates)
    raise FileNotFoundError(
        f"Official QFF2026_QML_Challenge_Kit not found.\n"
        f"Searched locations:\n{searched_str}\n"
        f"Set the QFF_CHALLENGE_KIT environment variable to the challenge kit directory."
    )


def get_dataset_dir(problem: str = "q1") -> Path:
    """Returns the path to the specified problem dataset directory."""
    kit = get_challenge_kit_path()
    data_dir = kit / "data" / problem
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found for problem '{problem}': {data_dir}")
    return data_dir


def get_tools_dir() -> Path:
    """Returns the path to the official tools directory."""
    tools_dir = get_challenge_kit_path() / "tools"
    if not tools_dir.exists():
        raise FileNotFoundError(f"Official tools directory not found: {tools_dir}")
    return tools_dir


def get_benchmark_dir() -> Path:
    """Returns the path to the official benchmark directory."""
    bm_dir = get_challenge_kit_path() / "benchmark"
    if not bm_dir.exists():
        raise FileNotFoundError(f"Official benchmark directory not found: {bm_dir}")
    return bm_dir


def get_noise_model_dir() -> Path:
    """Returns the path to the official noise model directory."""
    nm_dir = get_challenge_kit_path() / "noise_model"
    if not nm_dir.exists():
        raise FileNotFoundError(f"Official noise model directory not found: {nm_dir}")
    return nm_dir

