# src/tiling/partition.py
"""
Overlapping grid partitioner with a Partition-of-Unity (PoU) weighting,
used to split the varifold resampling problem into independent local tiles.
"""

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class Tile:
    """One local sub-problem produced by the partitioner."""
    multi: tuple          # grid multi-index, e.g. (ix, iy, iz)
    idx: np.ndarray       # int64 indices into the global point array (members of this tile)
    weight: np.ndarray    # float64 PoU weight psi_l(x_i) for each member, in (0, 1]
    core_lo: np.ndarray   # (3,) lower corner of the core cell  (normalised coords)
    core_hi: np.ndarray   # (3,) upper corner of the core cell


@dataclass
class HardTile:
    """
    One local sub-problem for the *hard* (non-overlapping) partition.

    Cores are disjoint and tile the domain exactly: every source point belongs to
    exactly one tile's `core_idx`. `opt_idx` additionally includes a halo ring
    (core expanded by `halo`) used at full weight only as boundary CONTEXT during
    optimisation, so representatives near the core edge are pulled by neighbouring
    source mass instead of collapsing inward.
    """
    multi: tuple          # grid multi-index
    core_idx: np.ndarray  # int64 points whose core cell is this tile (DISJOINT across tiles)
    opt_idx: np.ndarray   # int64 points in core + halo (optimisation target; OVERLAPS neighbours)
    core_lo: np.ndarray   # (3,) lower corner of the core cell
    core_hi: np.ndarray   # (3,) upper corner of the core cell


class GridPartitioner:
    """
    Partition a point cloud into overlapping tiles on a regular grid.

    Parameters
    ----------
    n_tiles : tuple[int, int, int]
        Number of core cells along (x, y, z). Use 1 on an axis to leave it
        unsplit (e.g. (4, 4, 1) tiles the xy-plane and keeps z whole -- this is
        the "coronal-slice" validation setting from the paper).
    overlap : float
        Width of the overlap band in the SAME units as the coordinates handed
        to `partition` (here: normalised [0, 1] coords). A good default is
        ~3 * sigma so that core-boundary representatives still see neighbouring
        source mass within the kernel's effective support.
    """

    def __init__(self, n_tiles, overlap):
        self.n_tiles = tuple(int(n) for n in n_tiles)
        self.overlap = float(overlap)

    # ---- per-axis Partition-of-Unity ramp --------------------------------
    def _axis_weight(self, x, lo, hi, dmin, dmax):
        """
        Trapezoidal weight along one axis for the core cell [lo, hi]:
          1 in the interior, ramps linearly to 0 across `overlap`-wide bands
          centred on each *interior* cell boundary, and 0 outside core +/- h.
        At a domain boundary (lo==dmin or hi==dmax) there is no ramp.
        """
        ov = self.overlap
        h = 0.5 * ov
        w = np.ones_like(x, dtype=np.float64)

        if lo > dmin + 1e-9:                       # interior boundary on the left
            w = np.minimum(w, np.clip((x - (lo - h)) / ov, 0.0, 1.0))
        if hi < dmax - 1e-9:                       # interior boundary on the right
            w = np.minimum(w, np.clip(((hi + h) - x) / ov, 0.0, 1.0))

        # crisp membership: drop points fully outside the halo [lo-h, hi+h]
        w[(x < lo - h) | (x > hi + h)] = 0.0
        return w

    # ---- main entry point -------------------------------------------------
    def partition(self, X):
        """
        Parameters
        ----------
        X : (N, D) array of coordinates (D == len(n_tiles), typically 3).

        Returns
        -------
        tiles : list[Tile]      non-empty tiles
        edges : list[np.ndarray]  per-axis cell edges (for plotting / debugging)
        """
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        assert D == len(self.n_tiles), "n_tiles rank must match coordinate dim"

        dmin = X.min(axis=0)
        dmax = X.max(axis=0)
        edges = [np.linspace(dmin[a], dmax[a], self.n_tiles[a] + 1) for a in range(D)]

        # PoU is only valid when the overlap band is narrower than a core cell;
        # otherwise a tile's own left/right ramps overlap and weights stop
        # summing to 1. Only split axes (n_tiles > 1) have interior boundaries.
        for a in range(D):
            if self.n_tiles[a] > 1:
                cell_w = (dmax[a] - dmin[a]) / self.n_tiles[a]
                if self.overlap >= cell_w:
                    raise ValueError(
                        f"overlap={self.overlap} must be < core cell width "
                        f"{cell_w:.4f} on axis {a} (n_tiles={self.n_tiles[a]}). "
                        f"Use fewer tiles on that axis or a smaller overlap.")

        tiles = []
        for multi in itertools.product(*[range(n) for n in self.n_tiles]):
            w = np.ones(N, dtype=np.float64)
            lo = np.empty(D)
            hi = np.empty(D)
            for a, k in enumerate(multi):
                lo[a], hi[a] = edges[a][k], edges[a][k + 1]
                w *= self._axis_weight(X[:, a], lo[a], hi[a], dmin[a], dmax[a])

            mask = w > 1e-8
            if not mask.any():
                continue
            tiles.append(Tile(multi=multi,
                              idx=np.nonzero(mask)[0].astype(np.int64),
                              weight=w[mask],
                              core_lo=lo.copy(),
                              core_hi=hi.copy()))
        return tiles, edges

    # ---- hard (non-overlapping) partition ---------------------------------
    def partition_hard(self, X, halo=None):
        """
        Disjoint core cells (each point assigned to exactly one) plus a halo ring
        used only as boundary context. No PoU weighting.

        Parameters
        ----------
        X : (N, D) coordinates.
        halo : float or None
            Halo width added around each core on every side (same units as X).
            Defaults to `self.overlap`. Unlike the PoU `partition`, the halo may
            be arbitrarily large (no cell-width constraint).

        Returns
        -------
        tiles : list[HardTile]
        edges : list[np.ndarray]
        """
        halo = self.overlap if halo is None else float(halo)
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        assert D == len(self.n_tiles), "n_tiles rank must match coordinate dim"

        dmin = X.min(axis=0)
        dmax = X.max(axis=0)
        edges = [np.linspace(dmin[a], dmax[a], self.n_tiles[a] + 1) for a in range(D)]

        # Hard core-cell index of every point along every axis.
        cell = np.empty((N, D), dtype=np.int64)
        for a in range(D):
            cell[:, a] = np.clip(
                np.searchsorted(edges[a], X[:, a], side="right") - 1,
                0, self.n_tiles[a] - 1)

        tiles = []
        for multi in itertools.product(*[range(n) for n in self.n_tiles]):
            core_mask = np.all(cell == np.asarray(multi), axis=1)
            if not core_mask.any():
                continue
            lo = np.array([edges[a][multi[a]] for a in range(D)])
            hi = np.array([edges[a][multi[a] + 1] for a in range(D)])
            opt_mask = np.all((X >= lo - halo) & (X <= hi + halo), axis=1)
            tiles.append(HardTile(multi=multi,
                                  core_idx=np.nonzero(core_mask)[0].astype(np.int64),
                                  opt_idx=np.nonzero(opt_mask)[0].astype(np.int64),
                                  core_lo=lo,
                                  core_hi=hi))
        return tiles, edges


def allocate_M(tiles, point_mass, M_total, M_min=16):
    """
    Distribute the global representative-point budget M_total across tiles,
    proportional to each tile's PoU-weighted mass.

    Parameters
    ----------
    tiles : list[Tile]
    point_mass : (N,) array   per-point mass w_i (e.g. nu_X.sum(axis=1)).
    M_total : int             global target number of representatives.
    M_min : int               floor so tiny tiles still get a few points.

    Returns
    -------
    M_l : (len(tiles),) int array, clipped to [M_min, n_points_in_tile].
    """
    mass = np.array([(t.weight * point_mass[t.idx]).sum() for t in tiles], dtype=np.float64)
    share = mass / mass.sum()
    M_l = np.round(M_total * share).astype(int)
    M_l = np.maximum(M_l, M_min)
    n_pts = np.array([len(t.idx) for t in tiles])
    M_l = np.minimum(M_l, n_pts)
    return M_l, mass
