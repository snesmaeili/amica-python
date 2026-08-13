.. _api:

=============
API Reference
=============

This page documents the stable public API of :mod:`jamica`.

Most users will interact with one of two interfaces:

* :class:`jamica.Amica` for fitting AMICA directly on NumPy arrays.
* :func:`jamica.fit_ica` for MNE-Python workflows.

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

Functions
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   fit_ica

Low-level Solver
----------------

.. autofunction:: jamica.amica
