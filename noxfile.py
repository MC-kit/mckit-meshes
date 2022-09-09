"""Nox sessions.

See `Cjolowicz's article <https://cjolowicz.github.io/posts/hypermodern-python-03-linting>`_
"""
from typing import Final, List

import re
import shutil
import sys

from glob import glob
from pathlib import Path
from textwrap import dedent

import nox

from nox import Session, session  # mypy: ignore

nox.options.sessions = (
    # "safety",   # TODO dvp: check if 'safety' session is necessary, if yes, return it
    "pre-commit",
    "xdoctest",
    "tests",
    "docs-build",
)


NAME_RGX = re.compile(r'name\s*=\s*"(?P<package>[-_a-zA-Z]+)"')


def find_my_name() -> str:
    """Find this package name.

    Search the first row in pyproject.toml in format

        name = "<package>"

    and returns the <package> value.
    This allows to reuse the noxfile.py in similar projects
    without any changes.

    Returns:
        str: Current package found in pyproject.toml

    Raises:
        ValueError: if the pattern is not found.
    """
    with Path("pyproject.toml").open() as fid:
        for line in fid:
            res = NAME_RGX.match(line)
            if res is not None:
                return res["package"].replace("-", "_")
    msg = "Cannot find package name"
    raise ValueError(msg)


package: Final = find_my_name()
locations: Final = f"src/{package}", "src/tests", "noxfile.py", "docs/source/conf.py"

supported_pythons: Final = "3.8", "3.9", "3.10", "3.11"
black_pythons: Final = "3.10"
mypy_pythons: Final = "3.10"
lint_pythons: Final = "3.10"


def activate_virtualenv_in_precommit_hooks(s: Session) -> None:
    """Activate virtualenv in hooks installed by pre-commit.

    This function patches git hooks installed by pre-commit to activate the
    session's virtual environment. This allows pre-commit to locate hooks in
    that environment when invoked from git.

    Args:
        s: The Session object.
    """
    virtualenv = s.env.get("VIRTUAL_ENV")
    if virtualenv is None:
        return

    hook_dir = Path(".git") / "hooks"
    if not hook_dir.is_dir():
        return

    for hook in hook_dir.iterdir():
        if hook.name.endswith(".sample") or not hook.is_file():
            continue

        text = hook.read_text()
        bin_dir = repr(s.bin)[1:-1]  # strip quotes
        if not (
            Path("A") == Path("a")
            and bin_dir.lower() in text.lower()
            or bin_dir in text
        ):
            continue

        lines = text.splitlines()
        if not (lines[0].startswith("#!") and "python" in lines[0].lower()):
            continue

        header = dedent(
            f"""\
            import os
            os.environ["VIRTUAL_ENV"] = {virtualenv!r}
            os.environ["PATH"] = os.pathsep.join((
                {s.bin!r},
                os.environ.get("PATH", ""),
            ))
            """
        )

        lines.insert(1, header)
        hook.write_text("\n".join(lines))


@session(name="pre-commit", python="3.10")
def precommit(s: Session) -> None:
    """Lint using pre-commit."""
    args = s.posargs or ["run", "--all-files", "--show-diff-on-failure"]
    s.run(
        "poetry",
        "install",
        "--only",
        "pre_commit,style,isort,black,flake8",
        external=True,
    )
    s.run("pre-commit", *args)
    if args and args[0] == "install":
        activate_virtualenv_in_precommit_hooks(s)


# @session(python="3.10")
# def safety(s: Session) -> None:
#     """Scan dependencies for insecure packages."""
#     requirements = s.poetry.export_requirements()
#     s.install("safety")
#     s.run("safety", "check", "--full-report", f"--file={requirements}", *s.posargs)


@session(python=supported_pythons)
def tests(s: Session) -> None:
    """Run the test suite."""
    s.run(
        "poetry",
        "install",
        "--only",
        "main,test,xdoctest,coverage",
        external=True,
    )
    # s.install("pytest", "pygments", "coverage[toml]")
    try:
        s.run("coverage", "run", "--parallel", "-m", "pytest", *s.posargs)
    finally:
        if s.interactive:
            s.notify("coverage", posargs=[])


@session
def coverage(s: Session) -> None:
    """Produce the coverage report.

    To obtain html report run
        nox -rs coverage -- html
    """
    args = s.posargs or ["report"]

    s.install("coverage[toml]")

    if not s.posargs and any(Path().glob(".coverage.*")):
        s.run("coverage", "combine")

    s.run("coverage", *args)


# TODO dvp: check some strange errors on 3.8, 3.9 and slow install of pandas on 3.11
@session(python="3.10")
def typeguard(s: Session) -> None:
    """Runtime type checking using Typeguard."""
    s.run(
        "poetry",
        "install",
        "--only",
        "main,test,typeguard",
        external=True,
    )
    # s.install("pytest", "typeguard", "pygments")
    s.run("pytest", f"--typeguard-packages={package}", *s.posargs)


@session(python="3.10")
def isort(s: Session) -> None:
    """Organize imports."""
    search_patterns = [
        "*.py",
        f"src/{package}/*.py",
        "src/tests/*.py",
        "benchmarks/*.py",
        "profiles/*.py",
    ]
    files_to_process: List[str] = sum(
        (glob(p, recursive=True) for p in search_patterns), []
    )
    if files_to_process:
        s.run(
            "poetry",
            "install",
            "--only",
            "isort",
            external=True,
        )
        s.run(
            "pycln",
            "--check",
            "--diff",
            *files_to_process,
            external=True,
        )
        s.run(
            "isort",
            "--check",
            "--diff",
            *files_to_process,
            external=True,
        )


@session(python=black_pythons)
def black(s: Session) -> None:
    """Run black code formatter."""
    args = s.posargs or locations
    s.run(
        "poetry",
        "install",
        "--only",
        "black",
        external=True,
    )
    s.run("black", *args)


@session(python=lint_pythons)
def lint(s: Session) -> None:
    """Lint using flake8."""
    args = s.posargs or locations
    s.run(
        "poetry",
        "install",
        "--only",
        "flake8",
        external=True,
    )
    s.run("flake8", *args)


@session(python=mypy_pythons)
def mypy(s: Session) -> None:
    """Type-check using mypy."""
    args = s.posargs or ["src", "docs/source/conf.py"]
    s.run(
        "poetry",
        "install",
        "--only",
        "main,mypy",
        external=True,
    )
    s.run("mypy", *args)
    if not s.posargs:
        s.run("mypy", f"--python-executable={sys.executable}", "noxfile.py")


@session(python=supported_pythons)
def xdoctest(s: Session) -> None:
    """Run examples with xdoctest."""
    args = s.posargs or ["all"]
    s.run(
        "poetry",
        "install",
        "--only",
        "main,xdoctest",
        external=True,
    )
    s.run("python", "-m", "xdoctest", package, *args)


@session(name="docs-build", python="3.10")
def docs_build(s: Session) -> None:
    """Build the documentation."""
    args = s.posargs or ["docs/source", "docs/_build"]
    s.run(
        "poetry",
        "install",
        "--only",
        "main,docs",
        external=True,
    )
    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    s.run("sphinx-build", *args)


@session(python="3.10")
def docs(s: Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    args = s.posargs or ["--open-browser", "docs/source", "docs/_build"]
    s.run(
        "poetry",
        "install",
        "--only",
        "main,docs,docs_auto",
        external=True,
    )
    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    s.run("sphinx-autobuild", *args)
