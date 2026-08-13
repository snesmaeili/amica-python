.. _examples:

Examples
========

Runnable, commented examples for **jamica**.

Run on your own data
--------------------

For EEG data, start from ``01_mne_integration.py`` to fit AMICA on an MNE raw object.

If your data is already a NumPy array with shape ``(n_channels, n_samples)``, use ``02_pure_jax_fitting.py``.

Installation
------------

From a local clone: ``pip install -e ".[all]"``

JAX uses an available GPU automatically when the compatible GPU package is installed; otherwise it runs on CPU.
