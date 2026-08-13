# amica has been renamed to [jamica](https://pypi.org/project/jamica/)

**Install `jamica` instead:**

```bash
pip install jamica
```

```python
from jamica import Amica, AmicaConfig, fit_ica
```

This `amica` release (0.1.1) is a pointer. It contains no modules of its own — it only depends on `jamica`, so an existing `pip install amica` still ends up with working code.

## Why the rename

The name `amica` installed a top-level `amica` module. So does [`amica-python`](https://pypi.org/project/amica-python/), an independent implementation of the same algorithm by another author. The two overwrote each other's files, so only one could exist in a given environment — the conda-forge packages even had to be declared mutually exclusive.

That is an awkward constraint for any user, and a particularly bad one here: this project's own benchmark suite compares the two implementations against each other.

Under the name `jamica` the two share no files and install side by side.

## Migrating

Every exported name kept its spelling, and `ica.method` is still `"amica"`. Migration is one line:

```python
from amica import Amica, AmicaConfig    # 0.1.0 and earlier
from jamica import Amica, AmicaConfig   # 0.2.0 onward
```

## Note on `import amica`

This release intentionally ships no `amica` module, so `import amica` will not resolve to this package — that namespace is left to `amica-python`. If you upgraded from 0.1.0 and your code does `import amica`, change the import to `jamica` as shown above.

Versions 0.0.1 and 0.1.0 remain on PyPI unchanged if you need to pin the historical package.

- Source: <https://github.com/snesmaeili/jamica>
- Documentation: <https://snesmaeili.github.io/jamica/>
