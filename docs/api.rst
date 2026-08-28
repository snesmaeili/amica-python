.. _api:

=============
API Reference
=============

This page documents the stable public API of :mod:`jamica`.

Most users will interact with one of these interfaces:

* :class:`jamica.Amica` for fitting AMICA directly on NumPy arrays.
* :func:`jamica.fit_ica` for single-model MNE-Python workflows.
* :class:`jamica.AmicaICA` for multi-model fits, which exposes one
  :class:`mne.preprocessing.ICA` per model.
* :func:`jamica.amica` as the stable single-model solver boundary for
  frameworks such as MNE that already whiten and PCA-reduce their data. See
  :doc:`mne_solver_contract` for its matrix and validation contract.

Core API
========

.. currentmodule:: jamica

Classes
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Amica
   AmicaConfig
   AmicaResult
   AmicaICA

Functions
---------

.. autofunction:: amica

.. autosummary::
   :toctree: generated/
   :nosignatures:

   fit_ica
   get_model_ica
   read_amica_ica

Warnings
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   JamicaConvergenceWarning
