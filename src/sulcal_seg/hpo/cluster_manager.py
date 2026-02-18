"""SLURM job script generation for distributed HPO."""
from pathlib import Path
from typing import Optional


class ClusterManager:
    """
    Generates SLURM job scripts for running HPO on a compute cluster.

    Two templates:
        - Single-node: Multiple trials on one node, shared GPU memory
        - Multi-node: Distributed Optuna workers across nodes via SQLite
    """

    def __init__(
        self,
        study_name: str,
        storage_path: str,
        conda_env: str = "sulcal_seg",
        partition: str = "gpu",
        output_dir: str = "slurm_logs",
    ) -> None:
        self.study_name = study_name
        self.storage_path = storage_path
        self.conda_env = conda_env
        self.partition = partition
        self.output_dir = output_dir

    def generate_single_node_script(
        self,
        n_trials: int = 20,
        n_gpus: int = 4,
        cpus_per_task: int = 8,
        mem_gb: int = 64,
        time_hours: int = 24,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a SLURM script for single-node HPO.

        Args:
            n_trials: Total trials to run on this node.
            n_gpus: Number of GPUs to request.
            cpus_per_task: CPUs per task.
            mem_gb: Total memory in GB.
            time_hours: Wall-clock time limit in hours.
            output_path: If given, write the script to this path.

        Returns:
            SLURM script as a string.
        """
        script = f"""#!/bin/bash
#SBATCH --job-name={self.study_name}_hpo
#SBATCH --partition={self.partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --mem={mem_gb}G
#SBATCH --time={time_hours:02d}:00:00
#SBATCH --output={self.output_dir}/{self.study_name}_%j.out
#SBATCH --error={self.output_dir}/{self.study_name}_%j.err

# Environment setup
module load cuda/11.8
source activate {self.conda_env}

# Run HPO
python -m sulcal_seg.experiments.run_ablations \\
    --study-name {self.study_name} \\
    --storage {self.storage_path} \\
    --n-trials {n_trials} \\
    --n-jobs {n_gpus}
"""
        if output_path is not None:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(script)
        return script

    def generate_multi_node_script(
        self,
        n_nodes: int = 4,
        trials_per_node: int = 10,
        n_gpus_per_node: int = 4,
        cpus_per_task: int = 8,
        mem_gb: int = 64,
        time_hours: int = 12,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a SLURM array script for multi-node distributed HPO.

        Each array task runs `trials_per_node` trials independently,
        all writing to the same SQLite database.

        Args:
            n_nodes: Number of SLURM array tasks (nodes).
            trials_per_node: Trials per array task.
            n_gpus_per_node: GPUs per array task.
            cpus_per_task: CPUs per task.
            mem_gb: Memory per task in GB.
            time_hours: Wall-clock time per task.
            output_path: If given, write the script to this path.

        Returns:
            SLURM array script as a string.
        """
        script = f"""#!/bin/bash
#SBATCH --job-name={self.study_name}_hpo_array
#SBATCH --partition={self.partition}
#SBATCH --array=0-{n_nodes - 1}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres=gpu:{n_gpus_per_node}
#SBATCH --mem={mem_gb}G
#SBATCH --time={time_hours:02d}:00:00
#SBATCH --output={self.output_dir}/{self.study_name}_node%a_%j.out
#SBATCH --error={self.output_dir}/{self.study_name}_node%a_%j.err

# Environment setup
module load cuda/11.8
source activate {self.conda_env}

echo "Array task ${{SLURM_ARRAY_TASK_ID}} starting HPO worker"

# Each node runs independently against shared SQLite storage
python -m sulcal_seg.experiments.run_ablations \\
    --study-name {self.study_name} \\
    --storage {self.storage_path} \\
    --n-trials {trials_per_node} \\
    --n-jobs {n_gpus_per_node} \\
    --worker-id ${{SLURM_ARRAY_TASK_ID}}
"""
        if output_path is not None:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(script)
        return script
