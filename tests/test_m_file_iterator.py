from __future__ import annotations

import pytest

from mckit_meshes.m_file_iterator import m_file_iterator


def test_file_with_single_mesh(data):
    fn = data / "1.m"
    with fn.open() as fid:
        it = m_file_iterator(fid)
        header = next(it)
        assert_is_header(header, fn, "323318560.00")
        mesh = next(it)
        assert_is_mesh_output(mesh)
        with pytest.raises(StopIteration):
            next(it)


def test_file_with_two_meshes(data):
    fn = data / "2.m"
    with fn.open() as fid:
        it = m_file_iterator(fid)
        header = next(it)
        assert_is_header(header, fn, "323318560.00")
        cnt = 0
        for mesh in it:
            assert_is_mesh_output(mesh)
            cnt += 1
        assert cnt == 2, "The file 2.m contains 2 meshes"


def assert_lines_are_trimmed(a_text):
    for line in a_text:
        assert line.strip() == line, "End of lines should be trimmed"


def assert_is_header(header, file_name, nps):
    assert_lines_are_trimmed(header)
    assert len(header) == 3, "The header is 3 lines long"
    assert header[0].startswith("mcnp"), 'The header starts with the word "mcnp"'
    assert nps in header[-1], f"The file {file_name} has nps == {nps}"


def assert_is_mesh_output(mesh):
    assert_lines_are_trimmed(mesh)
    assert_has_no_empty_lines(mesh)
    assert mesh[0].startswith(
        "Mesh Tally Number",
    ), 'The first line should start with "Mesh Tally Number"'
    assert mesh[-1].startswith(
        "2.000E+01     7.500     7.500     7.500",
    ), 'The last line should start with "2.000E+01     7.500     7.500     7.500"'


def assert_has_no_empty_lines(a_text):
    for line in a_text:
        assert len(line) > 0, "The text does not contain empty lines"
        assert line != "\n", "The text does not contain empty lines"
