# AI assistance in this project

Parts of jamica were written with the help of AI coding assistants. This page
says where, and — more importantly — what that does and does not mean for the
numbers the package produces.

## What was assisted

AI assistance was used for implementation, refactoring, test scaffolding,
documentation and tooling. Commits carrying a `Co-authored-by:` trailer naming
an assistant are the ones where it contributed directly; `git log` is the
authoritative record, and it is more precise than any prose summary could be.

The algorithm itself is not an AI contribution. AMICA is Palmer's, and the
reference behaviour jamica targets is the Fortran AMICA 1.7 implementation.

## What that does not change

Correctness here does not rest on how the code was typed. It rests on the
package reproducing an independent reference implementation on fixtures whose
expected values were not produced by this codebase:

- Single-model fits are checked against Fortran AMICA 1.7. Scope and known
  limits are in the README's Validation section; the protocols and the patched
  reference build are in [jamica-benchmark](https://github.com/snesmaeili/jamica-benchmark).
- The test suite runs on every supported Python and on both backends in CI.

An assistant that writes a plausible-looking but wrong update rule fails those
checks the same way a human writing the same mistake would. That is the point
of holding a numerical package to a reference rather than to a review of its
diffs.

## What a human is accountable for

Every change was reviewed and merged by a maintainer, who is responsible for
it regardless of how it was drafted. AI assistants are not authors: they are
not credited in `CITATION.cff`, they do not appear in the manuscript's author
list, and they cannot take responsibility for the work. Report problems on the
[issue tracker](https://github.com/snesmaeili/jamica/issues) and a maintainer
will answer for them.

## If you contribute

You are welcome to use AI assistance. Two requests:

1. Understand what you are submitting well enough to defend it in review.
1. Note the assistance, ideally as a `Co-authored-by:` trailer on the commit,
   so the record stays accurate.

Neither is a hurdle. The first is what code review already assumes; the second
just keeps provenance greppable.
