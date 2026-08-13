"""
AMICA via MNE-Python
====================

This example shows how to fit AMICA through the :func:`jamica.fit_ica`
wrapper.

The wrapper returns a standard :class:`mne.preprocessing.ICA` object, so the
usual MNE methods work as expected:

- ``plot_components``
- ``plot_sources``
- ``get_sources``
- ``apply``

The example uses the MNE sample dataset. To run it on your own data, replace
``load_raw()`` with your own loading code.
"""

# %%
# Imports
# -------

from __future__ import annotations

import mne

from jamica import fit_ica

# %%
# Load data
# ---------
#
# Here we use the MNE sample dataset. You can replace this function with your
# own data loader, for example:
#
# .. code-block:: python
#
#    raw = mne.io.read_raw_fif("your_data_raw.fif", preload=True)
#    raw = mne.io.read_raw_eeglab("your_data.set", preload=True)
#    raw = mne.io.read_raw_brainvision("your_data.vhdr", preload=True)


def load_raw() -> mne.io.BaseRaw:
    """Load and minimally preprocess a Raw recording."""
    sample = mne.datasets.sample.data_path()
    raw = mne.io.read_raw_fif(
        sample / "MEG" / "sample" / "sample_audvis_raw.fif",
        preload=True,
    )

    raw.pick("eeg")
    raw.filter(1.0, 40.0)

    return raw


raw = load_raw()

print(
    f"Raw data: {len(raw.ch_names)} EEG channels, "
    f"{raw.n_times} samples @ {raw.info['sfreq']:.0f} Hz"
)

# %%
# Fit AMICA
# ---------
#
# ``fit_ica`` fits AMICA and returns a standard MNE ``ICA`` object.

ica = fit_ica(
    raw,
    n_components=15,
    max_iter=500,
    random_state=42,
)

print(f"Fitted ICA with {ica.n_components_} components")

# %%
# Inspect sources
# ---------------
#
# Once fitted, the result behaves like any other MNE ICA object.

sources = ica.get_sources(raw)

print(f"Sources shape: {sources.get_data().shape}")

# %%
# Next steps
# ----------
#
# In an interactive session, you can use standard MNE visualization and
# cleaning methods:
#
# .. code-block:: python
#
#    ica.plot_components()
#    ica.plot_sources(raw)
#    ica.exclude = [0, 1]
#    ica.apply(raw)
