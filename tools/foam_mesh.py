"""Minimal READ-ONLY OpenFOAM mesh and boundary-field reader.

Enough to compute boundary surface integrals: face area vectors, and patch
values of scalar and vector fields. Handles ascii and binary, uniform and
nonuniform entries, plain faceList and binary faceCompactList.

Deliberately standalone — no OpenFOAM environment, no solver, no writes.
"""

import re
import struct
from pathlib import Path

import numpy as np

_LABEL = 4  # bytes; standard WM_LABEL_SIZE=32 builds


def _fmt(raw):
    m = re.search(rb"format\s+(\w+)", raw)
    return m.group(1).decode() if m else "ascii"


def _after_header(raw):
    """Index just past the FoamFile block."""
    i = raw.find(b"FoamFile")
    if i < 0:
        return 0
    depth, j = 0, raw.index(b"{", i)
    for k in range(j, len(raw)):
        if raw[k : k + 1] == b"{":
            depth += 1
        elif raw[k : k + 1] == b"}":
            depth -= 1
            if depth == 0:
                return k + 1
    return j


def read_points(path):
    raw = Path(path).read_bytes()
    binary = _fmt(raw) == "binary"
    i = _after_header(raw)
    m = re.compile(rb"(\d+)\s*\(").search(raw, i)
    n, body = int(m.group(1)), m.end()
    if binary:
        return np.frombuffer(raw[body : body + 24 * n], dtype="<f8").reshape(n, 3).copy()
    nums = re.findall(rb"\(([^()]*)\)", raw[body:])[:n]
    if len(nums) != n:
        raise ValueError(f"expected {n} points, parsed {len(nums)}")
    return np.array([[float(x) for x in v.split()] for v in nums])


def read_faces(path):
    """Return a list of point-index arrays, one per face."""
    raw = Path(path).read_bytes()
    binary = _fmt(raw) == "binary"
    compact = b"faceCompactList" in raw[: _after_header(raw)]
    i = _after_header(raw)
    if binary and compact:
        m = re.compile(rb"(\d+)\s*\(").search(raw, i)
        n_off, body = int(m.group(1)), m.end()
        offsets = np.frombuffer(raw[body : body + _LABEL * n_off], dtype="<i4")
        j = body + _LABEL * n_off
        m2 = re.compile(rb"(\d+)\s*\(").search(raw, j)
        n_v, body2 = int(m2.group(1)), m2.end()
        verts = np.frombuffer(raw[body2 : body2 + _LABEL * n_v], dtype="<i4")
        return [verts[offsets[k] : offsets[k + 1]] for k in range(n_off - 1)]
    if binary:
        raise NotImplementedError("binary non-compact faceList not supported")
    m = re.compile(rb"(\d+)\s*\(").search(raw, i)
    body = m.end()
    return [
        np.array([int(x) for x in g.split()])
        for g in re.findall(rb"\d+\(([^()]*)\)", raw[body:])
    ]


def read_boundary(path):
    """{name: {'type':..., 'nFaces':int, 'startFace':int}} in file order."""
    raw = Path(path).read_bytes().decode("latin-1")
    i = raw.find("}", raw.find("FoamFile")) + 1
    out = {}
    for m in re.finditer(r"\n\s{4}(\w[\w.]*)\s*\n\s{4}\{(.*?)\n\s{4}\}", raw[i:], re.S):
        body = m.group(2)
        def grab(key, cast=str, default=None):
            mm = re.search(rf"{key}\s+([^\s;]+)\s*;", body)
            return cast(mm.group(1)) if mm else default
        out[m.group(1)] = {
            "type": grab("type", str, "patch"),
            "nFaces": grab("nFaces", int, 0),
            "startFace": grab("startFace", int, 0),
        }
    return out


def face_area_vectors(points, faces):
    """Outward area vectors and centres, by fan decomposition about the mean.

    Sf = 1/2 sum_i (p_i - pbar) x (p_{i+1} - pbar), which is exact for planar
    polygons and is OpenFOAM's own construction for warped ones.
    """
    sf = np.zeros((len(faces), 3))
    cf = np.zeros((len(faces), 3))
    for k, idx in enumerate(faces):
        p = points[idx]
        pbar = p.mean(axis=0)
        q = p - pbar
        cross = np.cross(q, np.roll(q, -1, axis=0))
        sf[k] = 0.5 * cross.sum(axis=0)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        tri_c = (p + np.roll(p, -1, axis=0) + pbar) / 3.0
        total = area.sum()
        cf[k] = (tri_c * area[:, None]).sum(axis=0) / total if total > 0 else pbar
    return sf, cf


def _patch_blocks(raw):
    i = raw.find(b"boundaryField")
    if i < 0:
        return
    ms = list(re.finditer(rb"\n {4}([A-Za-z_][\w.]*)\n {4}\{", raw, re.M))
    ms = [m for m in ms if m.start() > i]
    for k, m in enumerate(ms):
        end = ms[k + 1].start() if k + 1 < len(ms) else len(raw)
        yield m.group(1).decode(), m.end(), end


def read_patch_field(path, n_components, n_faces_by_patch):
    """{patch: (n,) or (n,3) array} of boundary values for a written field."""
    raw = Path(path).read_bytes()
    binary = _fmt(raw) == "binary"
    kind = b"scalar" if n_components == 1 else b"vector"
    out = {}
    for name, start, end in _patch_blocks(raw):
        n = n_faces_by_patch.get(name, 0)
        m = re.compile(rb"nonuniform\s+List<" + kind + rb">\s*(\d+)\s*\(").search(
            raw, start, end
        )
        if m is not None:
            count, body = int(m.group(1)), m.end()
            if binary:
                buf = raw[body : body + 8 * n_components * count]
                arr = np.frombuffer(buf, dtype="<f8").copy()
            else:
                close = raw.index(b")", body) if n_components == 1 else end
                chunk = raw[body:close]
                if n_components == 1:
                    arr = np.array([float(x) for x in chunk.split()])
                else:
                    arr = np.array(
                        [[float(x) for x in g.split()]
                         for g in re.findall(rb"\(([^()]*)\)", chunk)[:count]]
                    ).ravel()
            out[name] = arr.reshape(count, n_components) if n_components > 1 else arr
            continue
        mu = re.compile(
            rb"uniform\s+(\(([^()]*)\)|[-\d.eE+]+)\s*;"
        ).search(raw, start, end)
        if mu is not None and n:
            if n_components == 1:
                out[name] = np.full(n, float(mu.group(1)))
            elif mu.group(2) is not None:
                vals = [float(x) for x in mu.group(2).split()]
                out[name] = np.tile(np.array(vals), (n, 1))
            # A vector patch written as `uniform <scalar>` is malformed for
            # this field; skip rather than crash, and let the caller report it
            # as a missing boundary value.
    return out


def read_labels(path):
    """A labelList (owner/neighbour), ascii or binary."""
    raw = Path(path).read_bytes()
    binary = _fmt(raw) == "binary"
    i = _after_header(raw)
    m = re.compile(rb"(\d+)\s*\(").search(raw, i)
    n, body = int(m.group(1)), m.end()
    if binary:
        return np.frombuffer(raw[body : body + _LABEL * n], dtype="<i4").copy()
    close = raw.index(b")", body)
    return np.array([int(x) for x in raw[body:close].split()])


def read_internal_field(path, n_components):
    """internalField of a written volField, ascii or binary, uniform or not."""
    raw = Path(path).read_bytes()
    binary = _fmt(raw) == "binary"
    kind = b"scalar" if n_components == 1 else b"vector"
    i = raw.find(b"internalField")
    m = re.compile(rb"nonuniform\s+List<" + kind + rb">\s*(\d+)\s*\(").search(raw, i)
    if m is None:
        mu = re.compile(rb"internalField\s+uniform\s+(\(([^()]*)\)|[-\d.eE+]+)\s*;").search(raw, i)
        if mu is None:
            return None
        if n_components == 1:
            return float(mu.group(1))
        return np.array([float(x) for x in mu.group(2).split()])
    n, body = int(m.group(1)), m.end()
    if binary:
        arr = np.frombuffer(raw[body : body + 8 * n_components * n], dtype="<f8").copy()
    else:
        chunk = raw[body : raw.index(b"\n)", body) + 1]
        if n_components == 1:
            arr = np.array([float(x) for x in chunk.split()])
        else:
            arr = np.array([[float(x) for x in g.split()]
                            for g in re.findall(rb"\(([^()]*)\)", chunk)[:n]]).ravel()
    return arr.reshape(n, n_components) if n_components > 1 else arr
