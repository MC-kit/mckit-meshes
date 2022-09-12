from typing import Generator, Iterator, List, TextIO

import datetime as dt
import re

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from numpy import ndarray as array

X, Y, Z = np.eye(3, dtype=float)
XY = np.vstack((X, Y))
XZ = np.vstack((X, Z))
YZ = np.vstack((Y, Z))
BASES = [XY, XZ, YZ]

FACTOR = 2.0 * 0.0005333332827654369
PLOTM_ORIGIN = np.array([1875.0, 1125.0], float)


@dataclass
class Page(object):
    lines: array
    basis: array
    origin: array
    extent: array
    date: dt.datetime = None
    title: str = None
    probid: dt.datetime = None
    rescaled: bool = None

    def __post_init__(self) -> None:
        assert self.lines.shape[1:] == (
            2,
            2,
        ), "Expecting array of pairs representing start and end points for a line"
        if not self.rescaled:
            res = np.asarray(self.lines, dtype=float)
            origin = self.get_2d_origin()
            extent = FACTOR * self.extent

            def apply(item):
                item -= PLOTM_ORIGIN
                item *= extent
                item += origin

            np.apply_along_axis(apply, -1, res)
            self.rescaled = True
            self.lines = res

    def convert_to_meters(self) -> "Page":
        lines = self.lines * 0.01
        return Page(
            lines=lines,
            basis=self.basis,
            origin=self.origin * 0.01,
            extent=self.extent * 0.01,
            date=self.date,
            title=self.title,
            probid=self.probid,
            rescaled=True,
        )

    def get_2d_origin(self):
        if self.basis is XY:
            return self.origin[:2]
        elif self.basis is XZ:
            return self.origin[0:3:2]
        elif self.basis is YZ:
            return self.origin[1:]
        else:
            raise RuntimeError("Only XY, XZ and YZ bases are supported for now")


def read(input_stream: TextIO) -> Generator[Page, None, None]:
    for page in load_pages(input_stream):
        yield transform_page(page)


def load(ps_file: Path) -> List[Page]:
    with ps_file.open() as fid:
        return list(read(fid))


def load_pages(input_stream: TextIO) -> Iterator[List[str]]:
    page = []
    for line in input_stream:
        if not line.startswith("%"):
            if line.endswith("showpage\n"):
                yield page
                page = []
            else:
                page.append(line)


def extract_description_lines(lines: List[str]) -> List[str]:
    description_lines = []
    for line in lines:
        if description_lines or line.startswith("     30   2205"):
            description_lines.append(line)
            if "extent = " in line:
                break
    return description_lines


def parse_description_lines(description_lines: List[str]):
    date = extract_date(description_lines[0])
    title = select_part_in_parenthesis(description_lines[1])
    line_no = 2
    while "probid" not in description_lines[line_no]:
        add_to_title = select_part_in_parenthesis(description_lines[line_no])
        title += " " + add_to_title
        line_no += 1
    probid = parse_us_date(select_part_in_parenthesis(description_lines[line_no])[10:])
    line_no += 2
    first_axis, second_axis = map(
        select_numbers, description_lines[line_no : line_no + 2]
    )
    basis = internalize_basis(np.vstack((first_axis, second_axis)))
    line_no += 3
    origin = select_numbers(description_lines[line_no])
    line_no += 1
    extent = select_numbers(description_lines[line_no])
    return date, title, probid, basis, origin, extent


def transform_page(
    page: List[str],
) -> Page:
    lines = collect_lines(page)
    description_lines = extract_description_lines(page[-20:])
    date, title, probid, basis, origin, extent = parse_description_lines(
        description_lines
    )
    return Page(lines, basis, origin, extent, date, title, probid)


def collect_lines(page: List[str]) -> array:
    lines = []
    for line in page[:-9]:
        line = line.split()
        if len(line) == 6:
            if line[2] == "moveto" and line[5] == "lineto":
                from_x, from_y = map(int, line[0:2])
                to_x, to_y = map(int, line[3:5])
                lines.append([[from_x, from_y], [to_x, to_y]])
    return np.array(lines, dtype=np.int32)


DOUBLE_PARENTHESIS_MATCHER = re.compile(r".*\(.*\\\((?P<numbers>.*)\\\)\).*")


def select_numbers(line: str) -> array:
    res = DOUBLE_PARENTHESIS_MATCHER.match(line)
    numbers = res.group("numbers")
    numbers = np.fromstring(numbers, dtype=float, sep=",")
    return numbers


def select_part_in_parenthesis(line: str) -> str:
    return line.split("(")[1].split(")")[0].strip()


def parse_us_date(string: str) -> dt.datetime:
    return dt.datetime.strptime(string, "%m/%d/%y %H:%M:%S")


def extract_date(line: str) -> dt.datetime:
    return parse_us_date(select_part_in_parenthesis(line))


def internalize_basis(basis: array) -> array:
    for b in BASES:
        if np.all(b == basis):
            return b
    return basis
