"""AMICA configuration dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


@dataclass
class AmicaConfig:
    """Configuration for AMICA algorithm.
    
    Parameters
    ----------
    num_models : int
        Number of ICA models to learn simultaneously. Default is 1.
    num_mix_comps : int
        Number of Gaussian mixture components per source. Default is 3.
    pcakeep : int, optional
        Number of PCA components to keep. If None, uses data rank.
    max_iter : int
        Maximum number of iterations. Default is 2000.
    lrate : float
        Initial learning rate. Default is 0.1.
    minlrate : float
        Minimum learning rate before stopping. Default is 1e-8.
    lratefact : float
        Factor to decrease learning rate on likelihood decrease. Default is 0.5.
    rholrate : float
        Learning rate for shape parameter rho. Default is 0.05.
    rholratefact : float
        Factor to decrease rho learning rate on likelihood decrease. Default is 0.5.
    minrho : float
        Minimum value for shape parameter (1.0 = Laplacian). Default is 1.0.
    maxrho : float
        Maximum value for shape parameter (2.0 = Gaussian). Default is 2.0.
    do_newton : bool
        Whether to use Newton updates for faster convergence. Default is True.
    newt_start : int
        Iteration to start Newton updates. Default is 50.
    newt_ramp : int
        Number of iterations to ramp up Newton. Default is 10.
    newtrate : float
        Newton learning rate multiplier. Default is 1.0.
    do_mean : bool
        Whether to remove data mean. Default is True.
    do_sphere : bool
        Whether to whiten/sphere data. Default is True.
    sphere_type : str
        Type of sphering: "pca" or "zca". Default is "pca".
    do_pca : bool
        Whether to apply PCA dimensionality reduction. Default is True.
    do_approx_sphere : bool
        Use approximate sphering. Default is True.
    mineig : float
        Minimum eigenvalue threshold for PCA. Default is 1e-12.
    do_reject : bool
        Whether to reject outlier samples. Default is False.
    rejsig : float
        Rejection threshold in standard deviations. Default is 3.0.
    rejstart : int
        Iteration to start rejection. Default is 2 (Klug et al. 2024).
    rejint : int
        Interval between rejection passes. Default is 3 (Klug et al. 2024).
    numrej : int
        Number of rejection passes per interval. Default is 5.
    min_dll : float
        Minimum log-likelihood change for convergence. Default is 1e-9.
    min_grad_norm : float
        Minimum gradient norm for convergence. Default is 1e-6.
    max_decs : int
        Number of LL decreases before reducing max learning rates. Default is 3.
    max_incs : int
        Number of small LL increases before stopping. Default is 10.
    use_grad_norm : bool
        Reserved for future use. Gradient norm convergence is not yet
        implemented in the solver. Default is False.
    use_min_dll : bool
        Whether to use min_dll for convergence. Default is True.
    invsigmax : float
        Maximum inverse sigma for numerical stability. Default is 100.0.
    invsigmin : float
        Minimum inverse sigma for numerical stability. Default is 0.0.
    doscaling : bool
        Whether to rescale A/mu/sbeta each iteration. Default is True.
    writestep : int
        Interval for writing intermediate results. Default is 100.
    outdir : Path, optional
        Output directory for results.
    fix_init : bool
        Use identity matrix initialization instead of random. Default is False.
    update_alpha : bool
        Whether to update mixture weights. Default is True.
    update_mu : bool
        Whether to update location parameters. Default is True.
    update_beta : bool
        Whether to update scale parameters. Default is True.
    update_rho : bool
        Whether to update shape parameters. Default is True.
    update_A : bool
        Whether to update mixing matrix. Default is True.
    update_c : bool
        Whether to update model centers. Default is True.
    update_gm : bool
        Whether to update model weights. Default is True.
    block_size : int
        Block size for stochastic updates. Default is 128.
    """
    
    # Model structure
    num_models: int = 1
    num_mix_comps: int = 3
    pcakeep: Optional[int] = None
    dtype: str = "float64"  # "float32" or "float64"
    
    # Iteration control
    max_iter: int = 2000
    
    # Learning rates
    lrate: float = 0.1
    minlrate: float = 1e-8
    lratefact: float = 0.5
    rholrate: float = 0.05
    rholratefact: float = 0.5
    
    # Shape parameter bounds
    minrho: float = 1.0
    maxrho: float = 2.0
    rho0: float = 1.5
    
    # Newton method
    do_newton: bool = True
    newt_start: int = 50
    newt_ramp: int = 10
    newtrate: float = 1.0
    
    # Preprocessing
    do_mean: bool = True
    do_sphere: bool = True
    sphere_type: str = "zca"
    do_pca: bool = True
    do_approx_sphere: bool = True
    mineig: float = 1e-12
    
    # Rejection
    do_reject: bool = False
    rejsig: float = 3.0
    rejstart: int = 2
    rejint: int = 3
    numrej: int = 5
    
    # Convergence
    min_dll: float = 1e-9
    min_grad_norm: float = 1e-6
    use_grad_norm: bool = False
    use_min_dll: bool = True
    max_decs: int = 3
    max_incs: int = 10
    
    # Numerical stability
    invsigmax: float = 100.0
    invsigmin: float = 0.0

    # Rescaling
    doscaling: bool = True
    
    # Output
    writestep: int = 100
    outdir: Optional[Path] = None
    
    # Initialization
    fix_init: bool = False
    
    # Update flags
    update_alpha: bool = True
    update_mu: bool = True
    update_beta: bool = True
    update_rho: bool = True
    update_A: bool = True
    update_c: bool = True
    update_gm: bool = True
    
    # Block processing
    block_size: int = 128
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_models < 1:
            raise ValueError("num_models must be >= 1")
        if self.num_mix_comps < 1:
            raise ValueError("num_mix_comps must be >= 1")
        if self.lrate <= 0:
            raise ValueError("lrate must be > 0")
        if self.minrho < 1.0:
            raise ValueError("minrho must be >= 1.0")
        if self.maxrho > 2.0:
            raise ValueError("maxrho must be <= 2.0")
        if self.minrho > self.maxrho:
            raise ValueError("minrho must be <= maxrho")
        if self.max_decs < 0:
            raise ValueError("max_decs must be >= 0")
        if self.max_incs < 0:
            raise ValueError("max_incs must be >= 0")
        if self.outdir is not None:
            self.outdir = Path(self.outdir)
