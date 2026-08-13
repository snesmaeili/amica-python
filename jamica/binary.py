"""
Binary runner for AMICA executable.
Wraps the standalone AMICA binary to run from Python using subprocess.

NOTE: This module is NOT usable without the external Fortran AMICA binary.
It is an optional wrapper provided for users who have compiled the original
binary independently. It is not guaranteed to be redistributable.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np

from .config import AmicaConfig
from .preprocessing import preprocess_data
from .solver import AmicaResult

logger = logging.getLogger(__name__)


class BinaryAmica:
    """AMICA runner using external binary executable.

    Parameters
    ----------
    binary_path : str or Path
        Path to the AMICA executable (e.g. 'amica15mkl.exe').
    config : AmicaConfig, optional
        Configuration parameters.
    keep_temp_files : bool
        If True, temporary files are not deleted after running.
    """

    def __init__(
        self,
        binary_path: str | Path,
        config: AmicaConfig | None = None,
        keep_temp_files: bool = False,
    ):
        self.binary_path = str(binary_path)
        self.config = config if config is not None else AmicaConfig()
        self.keep_temp_files = keep_temp_files

        if not os.path.exists(self.binary_path) and not shutil.which(self.binary_path):
            warnings.warn(f"AMICA binary not found at {self.binary_path}", stacklevel=2)

    def fit(self, data: np.ndarray, temp_dir: str | Path | None = None) -> AmicaResult:
        """Run AMICA binary on data.

        Parameters
        ----------
        data : np.ndarray, shape (n_channels, n_samples)
            Input data.
        temp_dir : str or Path, optional
            Directory to use for temporary files. If None, creates a system temp dir.

        Returns
        -------
        result : AmicaResult
            AMICA results.
        """
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("Data must be 2D (channels, samples)")

        # Create temp directory
        if temp_dir is None:
            temp_dir_obj = tempfile.TemporaryDirectory(prefix="amica_run_")
            work_dir = Path(temp_dir_obj.name)
        else:
            temp_dir_obj = None
            work_dir = Path(temp_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info("BinaryAmica: Preprocessing data...")
            data_white, mean, sphere, desphere, n_components, _ = preprocess_data(
                data,
                do_mean=self.config.do_mean,
                do_sphere=self.config.do_sphere,
                pcakeep=self.config.pcakeep,
                mineig=self.config.mineig,
                do_approx=self.config.do_approx_sphere,
                sphere_type=self.config.sphere_type,
            )

            # 2. Write Data File
            # AMICA expects float32 in Fortran (column-major) order.

            data_file = work_dir / "input.fdt"
            # Ensure float32
            data_white_f32 = data_white.astype(np.float32)
            # Write in Fortran order (column-major)
            data_white_f32.ravel(order="F").tofile(data_file)

            # 3. Write Param File
            param_file = work_dir / "input.param"
            self._write_param_file(
                param_file,
                data_file=data_file.name,
                out_dir="output",
                n_channels=n_components,
                n_samples=data.shape[1],
            )

            # Create output directory
            (work_dir / "output").mkdir(exist_ok=True)

            # 4. Run Binary
            cmd = [self.binary_path, "input.param"]
            logger.info("BinaryAmica: Running %s in %s", self.binary_path, work_dir)

            subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True, check=True)

            # 5. Load Results
            logger.info("BinaryAmica: Loading results...")
            res = self._load_results(work_dir / "output", n_components, data.shape[1])

            # Augment results with preprocessing info
            res.whitener_ = np.asarray(sphere)
            res.dewhitener_ = np.asarray(desphere)
            res.mean_ = np.asarray(mean)

            # Reconstruct sensor-space matrices with real preprocessing info
            A_white = res.mixing_matrix_white_
            W_white = res.unmixing_matrix_white_
            res.mixing_matrix_sensor_ = desphere @ A_white
            res.unmixing_matrix_sensor_ = W_white @ sphere

            return res

        finally:
            if temp_dir_obj and not self.keep_temp_files:
                temp_dir_obj.cleanup()

    def _write_param_file(
        self, fpath: Path, data_file: str, out_dir: str, n_channels: int, n_samples: int
    ):
        """Write AMICA input.param file.

        Parameters
        ----------
        fpath : Path
            Path to the parameter file to write.
        data_file : str
            Filename of the input data file.
        out_dir : str
            Directory name for AMICA output.
        n_channels : int
            Number of channels in the input data.
        n_samples : int
            Number of samples in the input data.
        """
        c = self.config

        # Determine flags for preprocessing
        # We did preprocessing in Python, so disable it in AMICA
        do_sphere = 0
        do_pca = 0
        do_mean = 0  # We removed mean

        # Param file content
        # Note: The keys are standard AMICA keywords.
        lines = [
            f"files {data_file}",
            f"outdir {out_dir}",
            f"num_chans {n_channels}",
            f"num_samples {n_samples}",
            f"num_models {c.num_models}",
            f"num_mix_comps {c.num_mix_comps}",
            f"max_iter {c.max_iter}",
            f"lrate {c.lrate}",
            f"minlrate {c.minlrate}",
            f"lratefact {c.lratefact}",
            f"rholrate {c.rholrate}",
            f"rholratefact {c.rholratefact}",
            f"minrho {c.minrho}",
            f"maxrho {c.maxrho}",
            f"do_newton {1 if c.do_newton else 0}",
            f"newt_start {c.newt_start}",
            f"newt_ramp {c.newt_ramp}",
            f"newtrate {c.newtrate}",
            f"do_reject {1 if c.do_reject else 0}",
            f"numrej {c.numrej}",
            f"rejsig {c.rejsig}",
            f"rejstart {c.rejstart}",
            f"writestep {c.writestep}",
            # Preprocessing disabled in binary
            f"do_mean {do_mean}",
            f"do_sphere {do_sphere}",
            f"do_pca {do_pca}",
            f"pcakeep {n_channels}",  # Already reduced if needed
        ]

        with open(fpath, "w") as f:
            f.write("\n".join(lines))

    def _load_results(self, out_dir: Path, n_components: int, n_samples: int) -> AmicaResult:
        """Load output files from AMICA output directory.

        Parameters
        ----------
        out_dir : Path
            Path to the directory containing AMICA output files.
        n_components : int
            Number of components fitted.
        n_samples : int
            Number of samples in the data.

        Returns
        -------
        result : AmicaResult
            The populated result object containing the fitted parameters.
        """

        def read_bin(name, shape):
            p = out_dir / name
            if not p.exists():
                return None
            data = np.fromfile(p, dtype=np.float64)
            data = data.reshape(shape, order="F")
            return data

        # NOTE: If we have multiple models, format changes. Assuming 1 model for now.
        if self.config.num_models > 1:
            raise NotImplementedError("Multi-model binary loading not yet implemented")

        W = read_bin("W", (n_components, n_components))
        if W is None:
            raise FileNotFoundError(f"Could not find W in {out_dir}")

        c = read_bin("c", (n_components,))
        if c is None:
            c = np.zeros(n_components)

        A = read_bin("A", (n_components, n_components))
        if A is None:
            A = np.linalg.pinv(W)

        S = read_bin("S", (n_components, n_components))
        if S is None:
            S = np.eye(n_components)

        n_mix = self.config.num_mix_comps
        alpha = read_bin("alpha", (n_mix, n_components))
        mu = read_bin("mu", (n_mix, n_components))
        beta = read_bin("beta", (n_mix, n_components))
        rho = read_bin("rho", (n_mix, n_components))

        ll_path = out_dir / "LL"
        if ll_path.exists():
            try:
                # Try loading as text
                LL = np.loadtxt(ll_path)
            except Exception:
                # Try binary
                LL = np.fromfile(ll_path, dtype=np.float64)
        else:
            LL = np.array([])

        gm = np.ones(1)  # Single model

        return AmicaResult(
            unmixing_matrix_white_=W,
            mixing_matrix_white_=A,
            unmixing_matrix_sensor_=W @ S,  # Populated later in run()
            mixing_matrix_sensor_=np.linalg.pinv(S) @ A,  # Populated later
            whitener_=S,
            dewhitener_=np.eye(n_components),  # Placeholder
            mean_=np.zeros(n_components),  # Placeholder
            alpha_=alpha if alpha is not None else np.zeros((n_mix, n_components)),
            mu_=mu if mu is not None else np.zeros((n_mix, n_components)),
            rho_=rho if rho is not None else np.zeros((n_mix, n_components)),
            sbeta_=beta if beta is not None else np.zeros((n_mix, n_components)),
            c_=c,
            gm_=gm,
            log_likelihood=LL,
            n_iter=len(LL),
            converged=False,
        )
