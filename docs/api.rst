.. _api:

=============
API Reference
=============

This page documents the stable public API of :mod:`jamica`.

Most users will interact with one of two interfaces:

* :class:`jamica.Amica` for fitting AMICA directly on NumPy arrays.
* :func:`jamica.fit_ica` for single-model MNE-Python workflows.
* :class:`jamica.AmicaICA` for multi-model fits, which exposes one
  :class:`mne.preprocessing.ICA` per model.

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

.. autosummary::
   :toctree: generated/
   :nosignatures:

   fit_ica
   get_model_ica
   read_amica_ica

Low-level Solver
----------------

.. autofunction:: jamica.amica
