#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["nox", "uv"]
# ///

"""Nox sessions for jamica development."""

from __future__ import annotations

import nox

nox.needs_version = ">=2025.10.14"
nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "tests"]


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.install("-e", ".[test]")
    session.run("pytest", *session.posargs)


@nox.session
def lint(session: nox.Session) -> None:
    """Run pre-commit checks."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session
def docs(session: nox.Session) -> None:
    """Build the documentation."""
    session.install("-e", ".[docs]")
    session.chdir("docs")
    session.run("make", "html", *session.posargs)


@nox.session
def build(session: nox.Session) -> None:
    """Build source and wheel distributions."""
    session.install("build")
    session.run("python", "-m", "build")


if __name__ == "__main__":
    nox.main()
