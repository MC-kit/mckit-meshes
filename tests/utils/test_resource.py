from __future__ import annotations

from pathlib import Path

import pytest

from mckit_meshes.utils.resource import path_resolver

THIS_FILENAME = Path(__file__).name


# noinspection PyCompatibility
@pytest.mark.parametrize(
    "package,resource,expected",
    [
        ("tests", "data/1.m", "data/1.m"),
    ],
)
def test_path_resolver(package, resource, expected) -> None:
    resolver = path_resolver(package)
    actual = resolver(resource)
    assert str(actual).replace("\\", "/").endswith(expected), "Failed to compute resource file name"
    assert Path(actual).exists(), f"The resource {resource!r} is not available"


@pytest.mark.parametrize(
    "package,resource",
    [
        ("tests", "data/not_existing"),
        ("mckit_meshes", "data/not_existing"),
    ],
)
def test_path_resolver_when_resource_doesnt_exist(package, resource) -> None:
    resolver = path_resolver(package)
    actual = resolver(resource)
    assert not Path(actual).exists(), f"The resource {resource!r} should not be available"


def test_path_resolver_when_package_doesnt_exist() -> None:
    with pytest.raises(ModuleNotFoundError):
        path_resolver("not_existing")("something.txt")


def test_path_resolver_local() -> None:
    resolver = path_resolver("tests")
    actual = resolver("utils/" + THIS_FILENAME)
    assert isinstance(actual, Path)
    assert actual.name == THIS_FILENAME
    assert actual.exists(), f"The file {THIS_FILENAME!r} should be available"


def test_path_resolver_in_own_package_with_separate_file() -> None:
    resolver = path_resolver("tests")
    assert resolver("__init__.py").exists(), "Should find '__init__.py' in the 'tests' package"


if __name__ == "__main__":
    pytest.main()
