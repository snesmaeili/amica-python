"""
Binary runner for AMICA executable.
Wraps the standalone AMICA binary to run from Python using subprocess.
"""
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Union, List, Tuple
import struct

import numpy as np

from .config import AmicaConfig
from .solver import AmicaResult
from .preprocessing import preprocess_data

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
        binary_path: Union[str, Path], 
        config: Optional[AmicaConfig] = None,
        keep_temp_files: bool = False
    ):
        self.binary_path = str(binary_path)
        self.config = config if config is not None else AmicaConfig()
        self.keep_temp_files = keep_temp_files
        
        if not os.path.exists(self.binary_path) and not shutil.which(self.binary_path):
            warnings.warn(f"AMICA binary not found at {self.binary_path}")

    def fit(self, data: np.ndarray, temp_dir: Optional[Union[str, Path]] = None) -> AmicaResult:
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
            # 1. Preprocess Data in Python (Mean remove, Sphere/PCA)
            # This ensures we control exact preprocessing parity with JAX backend
            # and avoids relying on AMICA binary's internal preprocessing options unless desired.
            
            print("BinaryAmica: Preprocessing data...")
            # We use the same preprocessing function as the JAX solver
            # We must map AMICA config to preprocessing kwargs
            data_white, mean, sphere, desphere, n_components, _ = preprocess_data(
                data,
                do_mean=self.config.do_mean,
                do_sphere=self.config.do_sphere,
                pcakeep=self.config.pcakeep,
                mineig=self.config.mineig,
                do_approx=self.config.do_approx_sphere,
                sphere_type=self.config.sphere_type
            )
            
            # 2. Write Data File
            # AMICA expects float32 typically
            # Flatten Fortran-style (column-major) or C-style?
            # MATLAB fwrite is column-major. numpy tofile is C-style (row-major).
            # So we need to transpose or use order='F'.
            # AMICA reads: channels x samples
            # Just to be safe with standard MATLAB/AMICA interoperability:
            # The standard is usually [ch1_s1, ch2_s1, ... chN_s1, ch1_s2, ...] (Interleaved)
            # OR [ch1_s1, ch1_s2, ... ch1_sM, ch2_s1 ...] (Block)
            # MATLAB fwrite(fid, data, 'float32') writes in column-major order (Fortran).
            # So in memory it iterates down columns first (Channel 1, 2, ... N for Sample 1, then Sample 2)
            # Wait, MATLAB matrices are (rows, cols) = (chans, samples).
            # Column-major means: (0,0), (1,0), (2,0)... (all chans for sample 0), then (0,1)...
            # So it is time-interleaved (multiplexed scanning).
            # We should write data.T.flatten()?? 
            # If data is (chans, samples). 
            # Fortran order of (chans, samples) is:
            # (0,0), (1,0), (2,0)... first column (first sample)
            # So the file is [s1_c1, s1_c2, ... s1_cN, s2_c1 ...]
            
            data_file = work_dir / "input.fdt"
            # Ensure float32
            data_white_f32 = data_white.astype(np.float32)
            # Write in Fortran order (column-major)
            data_white_f32.ravel(order='F').tofile(data_file)
            
            # 3. Write Param File
            param_file = work_dir / "input.param"
            self._write_param_file(
                param_file, 
                data_file=data_file.name, # Relative path usually safer if Cwd is work_dir
                out_dir="output",         # Subdir for output
                n_channels=n_components,  # Input to AMICA is already reduced/sphered
                n_samples=data.shape[1]
            )
            
            # Create output directory
            (work_dir / "output").mkdir(exist_ok=True)
            
            # 4. Run Binary
            cmd = [self.binary_path, "input.param"]
            print(f"BinaryAmica: Running {self.binary_path} in {work_dir}")
            
            # Use subprocess
            # Ensure no spaces in paths? work_dir might have them. 
            # Passing cwd=work_dir is best practice.
            result = subprocess.run(
                cmd,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                check=True
            )
            
            # Check stdout for errors or convergence info
            # print(result.stdout)
            
            # 5. Load Results
            print("BinaryAmica: Loading results...")
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

    def _write_param_file(self, fpath: Path, data_file: str, out_dir: str, n_channels: int, n_samples: int):
        """Write AMICA input.param file."""
        c = self.config
        
        # Determine flags for preprocessing
        # We did preprocessing in Python, so disable it in AMICA
        do_sphere = 0
        do_pca = 0
        do_mean = 0 # We removed mean
        
        # If we didn't sphere/PCA in python (e.g. config says no), we might want AMICA to do it?
        # But for 'Parity' and 'Fastest/Cleanest' path, doing it in Python is better.
        # Assuming we always do what config says in Python step.
        
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
            f"pcakeep {n_channels}", # Already reduced if needed
            # Shared components (optional, defaults to 0 usually)
            # block_size? AMICA binary has minimal batching control usually, or uses 'histbins'?
            # Check if block_size is supported. Usually it's automatic or hardcoded.
            # We can omit it.
        ]
        
        with open(fpath, "w") as f:
            f.write("\n".join(lines))

    def _load_results(self, out_dir: Path, n_components: int, n_samples: int) -> AmicaResult:
        """Load output files from AMICA output directory."""
        
        # Helper to read binary double files
        def read_bin(name, shape):
            p = out_dir / name
            if not p.exists():
                return None
            # AMICA writes doubles (float64)
            # Fortran order?
            # Standard W is square matrix
            data = np.fromfile(p, dtype=np.float64)
            data = data.reshape(shape, order='F') # MATLAB/Fortran order
            return data

        # Load W (Unmixing)
        # File is 'W' or 'W_1'...?
        # If num_models=1, it's just 'W' usually.
        # But let's check. runamica15.m loads 'W'.
        
        # NOTE: If we have multiple models, format changes. Assuming 1 model for now.
        if self.config.num_models > 1:
            raise NotImplementedError("Multi-model binary loading not yet implemented")

        W = read_bin("W", (n_components, n_components))
        if W is None:
            raise FileNotFoundError(f"Could not find W in {out_dir}")
        
        # Load c (centers)
        c = read_bin("c", (n_components,))
        if c is None: c = np.zeros(n_components)
        
        # Load A (Mixing)
        # AMICA usually writes 'A'
        A = read_bin("A", (n_components, n_components))
        # If not present, invert W
        if A is None:
            A = np.linalg.pinv(W)
            
        # Load sphere? We disabled sphering in binary, so S is Identity.
        # But if AMICA did it, it would be 'S'.
        S = read_bin("S", (n_components, n_components))
        if S is None:
            S = np.eye(n_components)
            
        # Load mixture params
        n_mix = self.config.num_mix_comps
        alpha = read_bin("alpha", (n_mix, n_components))
        mu = read_bin("mu", (n_mix, n_components))
        beta = read_bin("beta", (n_mix, n_components)) # often just 'beta'
        rho = read_bin("rho", (n_mix, n_components)) 
        
        # Load likelihood
        # File 'LL' usually text file? Or binary?
        # runamica15.m: load(fullfile(outdir, 'LL')) -> implies text if load() works, or binary?
        # Typically text for LL.
        ll_path = out_dir / "LL"
        if ll_path.exists():
            try:
                # Try loading as text
                LL = np.loadtxt(ll_path)
            except:
                # Try binary
                LL = np.fromfile(ll_path, dtype=np.float64)
        else:
            LL = np.array([])
            
        gm = np.ones(1) # Single model
        
        return AmicaResult(
            unmixing_matrix_white_=W,
            mixing_matrix_white_=A,
            unmixing_matrix_sensor_=W @ S,  # Populated later in run()
            mixing_matrix_sensor_=np.linalg.pinv(S) @ A,  # Populated later
            whitener_=S,
            dewhitener_=np.eye(n_components),  # Placeholder
            mean_=np.zeros(n_components),      # Placeholder
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
