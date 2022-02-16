from typing import Generator, List, TextIO


def m_file_iterator(stream: TextIO) -> Generator[List[str], None, None]:
    header: List[str] = []
    for line in stream:
        if len(header) < 3:
            line = line.strip()
            header.append(line)
        else:
            break
    yield header
    mesh: List[str] = []
    for line in stream:
        line = line.strip()
        if line:
            if len(mesh) > 0 and line.startswith("Mesh Tally Number"):
                yield mesh
                mesh = []
            mesh.append(line)
    if len(mesh) > 0:
        yield mesh
