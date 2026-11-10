"""Measured P1→P2 vector readouts shared by the desktop window and the browser.

Both front ends turn two picked atoms into the same physical displacement and
the same readout lines; each only decides how to color and place those lines.
A line tagged with a grain is drawn in that grain's current color.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crystal import (
    CrystalVectorCoordinates,
    axial_displacement,
    bicrystal,
    crystal_vector_coordinates,
    format_direction_components,
    layer_name,
    rotation_matrix_2d,
)

# Heights come from integer half-indices, so equal heights difference to an
# exact zero. This only guards floating heterophase layer heights.
AXIAL_SPLIT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class VectorReadoutLine:
    """One readout line. ``grain`` (0 or 1) colors it with that grain's color."""

    text: str
    grain: int | None = None


@dataclass(frozen=True)
class MeasuredVector:
    """A P1→P2 displacement with its full and in-plane grain representations.

    ``full`` and ``in_plane`` pair each reporting grain with the coordinates of
    the complete 3D displacement and of its in-plane part ``(x, y, 0)``; both
    use that grain's own frames and lattice constant. Same-grain picks report
    that grain; cross-grain picks report both grains.
    """

    crystal: object
    first: object
    second: object
    axial_repeat: int
    displacement: np.ndarray
    full: tuple[tuple[int, CrystalVectorCoordinates], ...]
    in_plane: tuple[tuple[int, CrystalVectorCoordinates], ...]

    @property
    def cross_grain(self) -> bool:
        return self.first.grain_index != self.second.grain_index

    @property
    def split(self) -> bool:
        """True when P1 and P2 differ in height, so the in-plane part differs."""
        return abs(float(self.displacement[2])) > AXIAL_SPLIT_TOLERANCE

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.displacement))

    @property
    def in_plane_length(self) -> float:
        return float(np.linalg.norm(self.displacement[:2]))

    @property
    def strained(self) -> bool:
        return any(coordinates.strained for _, coordinates in self.full)


def atom_name(atom) -> str:
    """``G1-A`` style grain and layer name of a selected atom."""
    return f"G{atom.grain_index + 1}-{layer_name(atom.layer)}"


def selected_vector_displacement(crystal, first, second, axial_repeat=0) -> np.ndarray:
    """Actual (x, y, z)/a0 from P1 to P2, with phase height and P2's periodic image.

    Both grains share the tilt axis even after their in-plane F and
    translations, so only z may be obtained from reference indices (each
    grain's own layer heights); x/y must come from the actual atom positions.
    """
    dz = axial_displacement(
        crystal, first.grain_index, first.half_indices,
        second.grain_index, second.half_indices, axial_repeat,
    )
    return np.r_[np.asarray(second.position, dtype=float) - np.asarray(first.position, dtype=float), dz]


def measure_vector(crystal, first, second, angle_deg, deformations, axial_repeat=0) -> MeasuredVector:
    """Express the P1→P2 displacement, and its in-plane part, in each grain."""
    crystal = bicrystal(crystal)
    displacement = selected_vector_displacement(crystal, first, second, axial_repeat)
    planar = np.r_[displacement[:2], 0.0]
    grains = (0, 1) if first.grain_index != second.grain_index else (first.grain_index,)

    def coordinates(vector, grain):
        return crystal_vector_coordinates(
            vector, (1 if grain == 0 else -1) * angle_deg / 2, deformations[grain],
            crystal.lattices[grain], crystal.axes[grain], scale=crystal.scales[grain],
        )

    return MeasuredVector(
        crystal, first, second, int(axial_repeat), displacement,
        tuple((grain, coordinates(displacement, grain)) for grain in grains),
        tuple((grain, coordinates(planar, grain)) for grain in grains),
    )


def vector_readout_lines(measured: MeasuredVector, *, display_rotation_deg=None) -> list[VectorReadoutLine]:
    """Readout lines for a measured vector.

    When P1 and P2 differ in height, every grain representation lists the full
    vector and its in-plane component separately, and the length line adds the
    in-plane length and the axial component. ``display_rotation_deg`` adds the
    displayed cross-grain x/y components; ``None`` omits that line.
    """
    crystal = measured.crystal
    kind = "Cross-grain" if measured.cross_grain else "Crystal"
    lines = [VectorReadoutLine(
        f"{kind} P1→P2 · {atom_name(measured.first)} → {atom_name(measured.second)}"
        f" · axial {measured.axial_repeat:+d}"
    )]
    split = measured.split

    def representation(label, attribute, unit_for):
        for (grain, full), (_, planar) in zip(measured.full, measured.in_plane, strict=True):
            unit = unit_for(grain)
            if split:
                lines.append(VectorReadoutLine(
                    f"G{grain + 1} {label} · full: "
                    f"{format_direction_components(getattr(full, attribute), unit)}", grain))
                lines.append(VectorReadoutLine(
                    f"G{grain + 1} {label} · in-plane: "
                    f"{format_direction_components(getattr(planar, attribute), unit)}", grain))
            else:
                lines.append(VectorReadoutLine(
                    f"G{grain + 1} {label}: "
                    f"{format_direction_components(getattr(full, attribute), unit)}", grain))

    # Each grain's components use its own lattice constant.
    representation("current (polar)", "current",
                   lambda grain: "a₀" if grain == 0 or crystal.scale == 1.0 else "a₀(G2)")
    # Material coordinates can stay integer while current spatial components
    # change under strain. Keep these notions explicitly apart.
    if measured.strained:
        representation("lattice [uvw]", "lattice", lambda grain: "")
    if split:
        lines.append(VectorReadoutLine(
            f"Current |Δr|/a₀ = {measured.length:.4f}"
            f" · in-plane |Δxy|/a₀ = {measured.in_plane_length:.4f}"
            f" · Δz/a₀ = {float(measured.displacement[2]):+.4f}"
        ))
    else:
        lines.append(VectorReadoutLine(f"Current |Δr|/a₀ = {measured.length:.4f}"))
    if measured.cross_grain and display_rotation_deg is not None:
        projected = rotation_matrix_2d(display_rotation_deg) @ measured.displacement[:2]
        lines.append(VectorReadoutLine(f"View Δxy/a₀ = ({projected[0]:.4f}, {projected[1]:.4f})"))
    return lines


def readout_text(lines) -> str:
    return "\n".join(line.text for line in lines)
