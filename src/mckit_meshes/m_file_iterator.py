from __future__ import annotations

from typing import Generator, TextIO


def m_file_iterator(stream: TextIO) -> Generator[list[str], None, None]:
    header: list[str] = []
    for _line in stream:
        if len(header) < 3:
            header.append(_line.strip())
        else:
            break
    yield header
    mesh: list[str] = []
    for line in stream:
        _line = line.strip()
        if _line:
            if len(mesh) > 0 and _line.startswith("Mesh Tally Number"):
                yield mesh
                mesh = []
            mesh.append(_line)
    if len(mesh) > 0:
        yield mesh
