from __future__ import annotations

from typing import TYPE_CHECKING, TextIO

import datetime as dt
import re

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from mckit_meshes.plot.check_coordinate_plane import BASES, is_x_plane, is_y_plane, is_z_plane

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from numpy.typing import NDArray


FACTOR = 2.0 * 0.0005333332827654369
PLOTM_ORIGIN = np.array([1875.0, 1125.0], float)


@dataclass
class Page:
    """Page of a plotm file."""

    lines: NDArray
    """Lines to plot"""
    basis: NDArray
    """Basis of plotm projection plane."""
    origin: NDArray
    """Projection plane center coordinates."""
    extent: NDArray
    """Extension of projection plane along basis axes."""
    date: dt.datetime
    """Time of the page computation."""
    title: str
    """Model title."""
    probid: dt.datetime
    """Time of the computation start."""
    rescaled: bool = False
    """Skip rescaling on initialization."""

    def __post_init__(self):
        assert self.lines.shape[1:] == (
            2,
            2,
        ), "Expecting array of pairs representing start and end points for a line"
        if not self.rescaled:
            res = np.asarray(self.lines, dtype=float)
            self.normalize_basis()
            origin = self.get_2d_origin()
            extent = FACTOR * self.extent

            def apply(item):
                item -= PLOTM_ORIGIN
                item *= extent
                item += origin

            np.apply_along_axis(apply, -1, res)
            self.rescaled = True
            self.lines = res

    def convert_to_meters(self) -> Page:
        """Convert measurement units from centimeters to meters."""
        return Page(
            lines=self.lines * 0.01,
            basis=self.basis,
            origin=self.origin * 0.01,
            extent=self.extent * 0.01,
            date=self.date,
            title=self.title,
            probid=self.probid,
            rescaled=True,
        )

    def is_x_plane(self):
        """Check, if the Page is in a plane perpendicular to X-axes."""
        return is_x_plane(self.basis)

    def is_y_plane(self):
        """Check, if the Page is in a plane perpendicular to Y-axes."""
        return is_y_plane(self.basis)

    def is_z_plane(self):
        """Check, if the Page is in a plane perpendicular to Z-axes."""
        return is_z_plane(self.basis)

    def normalize_basis(self) -> None:
        x_norm, y_norm = map(np.linalg.vector_norm, self.basis)
        if np.isclose(x_norm, 0.0) or np.isclose(y_norm, 0.0):
            raise ValueError
        if not np.isclose(x_norm, 1.0) and np.isclose(y_norm, 1.0):
            x_axis, y_axis = self.basis
            self.basis[0] = x_axis / x_norm
            self.basis[1] = y_axis / y_norm

    def get_2d_origin(self) -> NDArray:
        """Compute the coordinates of origin in projection plane.

        If the basis correspondes to one of coordinate planes or
        origin iz zero uses original numbers. Otherwise computes
        projection of `origin` to basis axes.
        This provides meaningful `x`, `y` coordinates on plots,
        corresponding to values in global coordinates.

        Returns
        -------
            coordinates to be base for line points
        """
        if self.is_z_plane():
            return self.origin[:2]
        if self.is_y_plane():
            return self.origin[0:3:2]
        if self.is_x_plane():
            return self.origin[1:]
        if np.all(self.origin == 0.0):
            return np.array([0.0, 0.0])
        x0 = np.dot(self.origin, self.basis[0])
        y0 = np.dot(self.origin, self.basis[1])
        return np.array([x0, y0])


def scan_pages(input_stream: TextIO) -> Iterator[Page]:
    """Iterate over pages defined in input stream.

    Parameters
    ----------
    input_stream
        read content of the plotm file

    Yields
    ------
        Pages from plotm file
    """
    for page in split_input(input_stream):
        yield transform_page(page)


def load_all_pages(ps_file: Path) -> list[Page]:
    """Load all the pages from a plotm file as a list of Pages.

    Parameters
    ----------
    ps_file
        path to plotm file

    Returns
    -------
        list of Pages
    """
    with ps_file.open() as fid:
        return list(scan_pages(fid))


def split_input(input_stream: TextIO) -> Iterator[list[str]]:
    """Split input from plotm file to text sections.

    Parameters
    ----------
    input_stream
        Stream with content of a plotm file.

    Yields
    ------
        list[str]: portion of input for one page
    """
    page: list[str] = []
    for line in input_stream:
        if not line.startswith("%"):
            if line.endswith("showpage\n"):
                yield page
                page = []
            else:
                page.append(line)


def _extract_description_lines(lines: list[str]) -> list[str]:
    description_lines: list[str] = []
    for line in lines:
        if description_lines or line.startswith("     30   2205"):
            description_lines.append(line)
            if "extent = " in line:
                break
    return description_lines


def _parse_description_lines(description_lines):
    date = _extract_date(description_lines[0])
    title = _select_part_in_parenthesis(description_lines[1])
    line_no = 2
    while "probid" not in description_lines[line_no]:
        add_to_title = _select_part_in_parenthesis(description_lines[line_no])
        title += " " + add_to_title
        line_no += 1
    probid = _parse_us_date(_select_part_in_parenthesis(description_lines[line_no])[10:])
    line_no += 2
    first_axis, second_axis = map(_select_numbers, description_lines[line_no : line_no + 2])
    basis = _internalize_basis(np.vstack((first_axis, second_axis)))
    line_no += 3
    origin = _select_numbers(description_lines[line_no])
    line_no += 1
    extent = _select_numbers(description_lines[line_no])
    return date, title, probid, basis, origin, extent


def transform_page(
    page: list[str],
) -> Page:
    lines = collect_lines(page)
    description_lines = _extract_description_lines(page[-20:])
    date, title, probid, basis, origin, extent = _parse_description_lines(description_lines)
    return Page(lines, basis, origin, extent, date, title, probid)


class CollectLinesState(IntEnum):
    """Enumerator for parsing  lines from a plotm file."""

    wait_line_width = 1
    reading_segments = 2


def collect_lines(section: list[str]) -> NDArray:
    """Collect lines to plot from a text section.

    Parameters
    ----------
    section
        text lines for a one page from a plotm file

    Returns
    -------
        array with (from, to) coordinates
    """
    red = False
    state = CollectLinesState.wait_line_width
    lines: list[list[list[int]]] = []
    segments: list[list[list[int]]] = []
    for _line in section[:-9]:
        line = _line.split()
        if line[-1] == "setrgbcolor":
            state = CollectLinesState.wait_line_width
            red = line[0] == "0.75"
        elif state == CollectLinesState.wait_line_width:
            if line[-1] == "setlinewidth":
                state = CollectLinesState.reading_segments  # start of new line
        elif state == CollectLinesState.reading_segments:
            if line[-1] == "stroke":
                if red:
                    segments = _squeeze_red_segments(segments)
                lines.extend(segments)
                segments = []
            elif len(line) == 6 and line[2] == "moveto" and line[5] == "lineto":
                from_x, from_y = map(int, line[0:2])
                to_x, to_y = map(int, line[3:5])
                segments.append([[from_x, from_y], [to_x, to_y]])
    return np.array(lines, dtype=np.int32)


def _squeeze_red_segments(segments: list[list[list[int]]]) -> list[list[list[int]]]:
    # fill gaps in dash lines
    fixed_segments: list[list[list[int]]] = []
    for segment in segments:
        if fixed_segments:
            prev_to = fixed_segments[-1][-1]
            next_from = segment[0]
            if prev_to != next_from:
                fixed_segments.append([prev_to, next_from])
        fixed_segments.append(segment)
    if len(fixed_segments) > 1:
        # reduce straight sections
        array_1 = np.array(fixed_segments, dtype=np.int32)
        all_x = array_1[:, :, 0].ravel()
        if _is_constant(all_x):  # vertical line
            return [[fixed_segments[0][0], fixed_segments[-1][-1]]]
        all_y = array_1[:, :, 1].ravel()
        if _is_constant(all_y):  # horizontal line
            return [[fixed_segments[0][0], fixed_segments[-1][-1]]]
        # Check if a single segment from begin to end fits the points defined with segements
        all_beg_x = array_1[:, 0, 0]
        beg_x = all_beg_x[0]
        end_x = array_1[-1, 1, 0]
        all_beg_y = array_1[:, 0, 1]
        beg_y = all_beg_y[0]
        end_y = array_1[-1, 1, 1]
        denominator = end_x - beg_x
        if np.abs(denominator) > 0:
            nominator = end_y - all_beg_y[0]
            m = nominator / denominator
            residuals = np.abs(all_beg_y[1:] - (m * (all_beg_x[1:] - beg_x) + beg_y))
            estimation = np.max(residuals) / np.hypot(denominator, nominator)
            if estimation < 1e-2:
                return [[fixed_segments[0][0], fixed_segments[-1][-1]]]

    return fixed_segments


def _is_constant(all_x: NDArray) -> bool:
    return bool(np.all(all_x[1:] == all_x[0]))


_DOUBLE_PARENTHESIS_MATCHER = re.compile(r".*\(.*\\\((?P<numbers>.*)\\\)\).*")


def _select_numbers(line: str) -> NDArray:
    res = _DOUBLE_PARENTHESIS_MATCHER.match(line)
    if res is None:
        msg = f"Cannot find number in line {line}"
        raise ValueError(msg)
    numbers = res.group("numbers")
    return np.fromstring(numbers, dtype=float, sep=",")


def _select_part_in_parenthesis(line: str) -> str:
    return line.split("(")[1].split(")")[0].strip()


def _parse_us_date(string: str) -> dt.datetime:
    return dt.datetime.strptime(string, "%m/%d/%y %H:%M:%S").replace(tzinfo=dt.UTC)


def _extract_date(line: str) -> dt.datetime:
    return _parse_us_date(_select_part_in_parenthesis(line))


def _internalize_basis(basis: NDArray) -> NDArray:
    for b in BASES:
        if np.all(b == basis):
            return b
    return basis
