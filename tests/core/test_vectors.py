"""Actual crystal-frame vectors and material coordinates under strain and rotation."""

import unittest
import numpy as np
from dichromatic_map import crystal
from .helpers import rotation


def basis(angle, f, lattice, axis):
    transform = np.eye(3)
    transform[:2, :2] = f @ rotation(angle)
    return transform @ crystal.get_geometry(lattice, axis).frame.T


class VectorPhysicsTests(unittest.TestCase):
    def test_sc_110_integer_translation_includes_axial_component(self):
        # SC conventional [2,-1,3] projects onto [-1,1,0]/sqrt(2),
        # [0,0,1], [1,1,0]/sqrt(2). Build this without the geometry helper.
        reference = np.array([2, -1, 3])
        angle = 19.0
        f = rotation(0.7) @ np.array([[1.03, 0.015], [0.015, 0.98]])
        planar = f @ rotation(angle) @ [-3 / np.sqrt(2), 3]
        displacement = np.r_[planar, 1 / np.sqrt(2)]
        value = crystal.crystal_vector_coordinates(
            displacement, angle, f, lattice="SC", axis="110"
        )
        np.testing.assert_allclose(value.lattice, reference, atol=1e-12)
        np.testing.assert_allclose(value.current_frame @ value.current,
                                   displacement, atol=1e-12)
        self.assertEqual(crystal.format_direction_components(value.lattice, unit=""),
                         "[2 -1 3]")
        self.assertTrue(value.strained)

    def test_unstrained_vector_has_two_crystal_representations(self):
        value = np.array([0.5, 0.5, 1.0])
        d = basis(30, np.eye(2), "BCC", "100") @ value
        first = crystal.crystal_vector_coordinates(d, 30, lattice="BCC", axis="100")
        second = crystal.crystal_vector_coordinates(d, -30, lattice="BCC", axis="100")
        np.testing.assert_allclose(first.current, value, atol=1e-12)
        self.assertGreater(np.linalg.norm(first.current - second.current), 0.1)
        for v in (first, second):
            np.testing.assert_allclose(v.current_frame @ v.current, d, atol=1e-12)
            np.testing.assert_allclose(v.lattice_basis @ v.lattice, d, atol=1e-12)
            np.testing.assert_allclose(v.current, v.lattice, atol=1e-12)
            self.assertFalse(v.strained)
        self.assertEqual(
            crystal.format_direction_components(first.current), "a₀/2[1 1 2]"
        )

    def test_strain_changes_current_components_not_material_indices(self):
        for lattice in ("FCC", "BCC", "SC"):
            for axis in ("100", "110", "111", "112", "1 -1 3"):
                u = np.array([[1.04, 0.009], [0.009, 0.98]])
                r = rotation(0.7)
                f = r @ u
                ref = np.array([0.5, 1.0, 1.5])
                d = basis(19, f, lattice, axis) @ ref
                v = crystal.crystal_vector_coordinates(d, 19, f, lattice, axis)
                np.testing.assert_allclose(v.lattice, ref, atol=1e-12)
                u3 = np.eye(3)
                u3[:2, :2] = u
                ref_frame = basis(19, np.eye(2), lattice, axis)
                expected = ref_frame.T @ u3 @ ref_frame @ ref
                np.testing.assert_allclose(v.current, expected, atol=1e-12)
                self.assertGreater(np.linalg.norm(v.current - ref), 1e-4)
                self.assertTrue(v.strained)
                self.assertAlmostEqual(
                    np.linalg.norm(v.current), np.linalg.norm(d), places=12
                )

    def test_rigid_rotation_alone_is_not_strain(self):
        ref = np.array([0.5, 0.5, 1.0])
        for angle in (0, 19, -25):
            f = rotation(1.1)
            d = basis(angle, f, "FCC", "110") @ ref
            v = crystal.crystal_vector_coordinates(d, angle, f)
            np.testing.assert_allclose(v.current, ref, atol=1e-12)
            np.testing.assert_allclose(v.lattice, ref, atol=1e-12)
            self.assertFalse(v.strained)

    def test_cross_grain_uses_both_endpoint_deformations_and_translations(self):
        fs = (
            rotation(0.6) @ np.array([[1.03, 0.008], [0.008, 0.98]]),
            rotation(-0.4) @ np.array([[0.99, -0.006], [-0.006, 1.02]]),
        )
        shifts = (np.array([0.3, -0.2, 0]), np.array([-0.1, 0.4, 0]))
        refs = (np.array([0.5, 0.5, 1]), np.array([2, 1, 0.5]))
        bs = [basis(a, f, "FCC", "112") for a, f in zip((17, -17), fs)]
        d = bs[1] @ refs[1] + shifts[1] - (bs[0] @ refs[0] + shifts[0])
        for g, angle in enumerate((17, -17)):
            v = crystal.crystal_vector_coordinates(d, angle, fs[g], "FCC", "112")
            np.testing.assert_allclose(v.lattice, np.linalg.solve(bs[g], d), atol=1e-12)
            np.testing.assert_allclose(v.current_frame @ v.current, d, atol=1e-12)
            self.assertGreater(np.linalg.norm(v.lattice - (refs[1] - refs[0])), 0.1)
            self.assertGreater(np.linalg.norm(v.current - v.lattice), 0.001)

    def test_format_preserves_length_and_does_not_invent_integer_indices(self):
        for vector, expected in (
            ([0.5, 0.5, 1], "a₀/2[1 1 2]"),
            ([-1, -1, -2], "a₀[-1 -1 -2]"),
            ([1.5, 1.5, 3], "3a₀/2[1 1 2]"),
            ([0, 0, 0], "a₀[0 0 0]"),
            ([1 / 3, 2 / 3, 0], "a₀/3[1 2 0]"),
        ):
            self.assertEqual(crystal.format_direction_components(vector), expected)
        self.assertEqual(
            crystal.format_direction_components([0.5, 0.5, 1], unit=""), "1/2[1 1 2]"
        )
        for vector in (
            [0.500001, 0.5, 1],
            [np.sqrt(2), 1, 0],
            [1 / 37, 1 / 41, 0],
            [257, 1, 0],
        ):
            self.assertTrue(
                crystal.format_direction_components(vector).startswith("≈ ")
            )
        for deformation in (np.zeros((2, 2)), np.diag([-1, 1]), [[1, 0], [0, np.nan]]):
            with self.assertRaises(ValueError):
                crystal.crystal_vector_coordinates([1, 2, 3], 0, deformation)


class ReadoutTests(unittest.TestCase):
    """Shared P1→P2 readout lines used by the desktop window and the browser."""

    def measure(self, second_position=(1.0, 0.5), second_half=(1, 0, 1), second_layer=1,
                second_grain=0, axial_repeat=0, deformations=None, lattice="FCC", axis="110",
                scale=1.0):
        from dichromatic_map.state import SelectedAtom
        from dichromatic_map.vectors import measure_vector
        first = SelectedAtom(np.zeros(2), 0, 0, np.zeros(3, dtype=int))
        second = SelectedAtom(np.array(second_position, dtype=float), second_grain,
                              second_layer, np.array(second_half))
        return measure_vector(crystal.bicrystal(lattice, axis, scale), first, second, 22.0,
                              deformations or (np.eye(2), np.eye(2)), axial_repeat)

    def test_same_height_endpoints_keep_a_single_representation(self):
        from dichromatic_map.vectors import readout_text, vector_readout_lines
        measured = self.measure((1.0, 0.0), (1, -1, 0), 0)
        self.assertFalse(measured.split)
        lines = vector_readout_lines(measured, display_rotation_deg=0.0)
        self.assertEqual([line.grain for line in lines], [None, 0, None])
        self.assertEqual(lines[0].text, "Crystal P1→P2 · G1-A → G1-A · axial +0")
        self.assertTrue(lines[1].text.startswith("G1 current (polar): "))
        self.assertEqual(lines[2].text, "Current |Δr|/a₀ = 1.0000")
        self.assertNotIn("in-plane", readout_text(lines))
        self.assertEqual(readout_text(lines), "\n".join(line.text for line in lines))

    def test_different_heights_list_full_and_in_plane_components(self):
        from dichromatic_map.vectors import vector_readout_lines
        measured = self.measure()
        self.assertTrue(measured.split)
        np.testing.assert_allclose(measured.displacement, [1, .5, np.sqrt(2) / 4], atol=1e-12)
        (_, full), = measured.full
        (_, planar), = measured.in_plane
        # The in-plane part removes the component along the cubic tilt axis [110].
        axis = np.array([1, 1, 0]) / np.sqrt(2)
        np.testing.assert_allclose(planar.current, full.current - (full.current @ axis) * axis, atol=1e-12)
        np.testing.assert_allclose(planar.lattice, planar.current, atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(planar.current), measured.in_plane_length, places=12)
        self.assertAlmostEqual(measured.in_plane_length, np.hypot(1, .5), places=12)
        lines = vector_readout_lines(measured)
        self.assertEqual([line.grain for line in lines], [None, 0, 0, None])
        self.assertEqual(lines[0].text, "Crystal P1→P2 · G1-A → G1-B · axial +0")
        self.assertEqual(lines[1].text, "G1 current (polar) · full: "
                         + crystal.format_direction_components(full.current))
        self.assertEqual(lines[2].text, "G1 current (polar) · in-plane: "
                         + crystal.format_direction_components(planar.current))
        self.assertEqual(lines[3].text,
                         "Current |Δr|/a₀ = 1.1726 · in-plane |Δxy|/a₀ = 1.1180 · Δz/a₀ = +0.3536")

    def test_axial_image_alone_splits_the_readout(self):
        from dichromatic_map.vectors import vector_readout_lines
        measured = self.measure((1.0, 0.0), (1, -1, 0), 0, axial_repeat=-1)
        self.assertTrue(measured.split)
        lines = vector_readout_lines(measured)
        self.assertEqual(lines[0].text, "Crystal P1→P2 · G1-A → G1-A · axial -1")
        (_, planar), = measured.in_plane
        self.assertEqual(lines[2].text, "G1 current (polar) · in-plane: "
                         + crystal.format_direction_components(planar.current))
        np.testing.assert_allclose(planar.current_frame @ planar.current, [1, 0, 0], atol=1e-12)
        self.assertIn("in-plane |Δxy|/a₀ = 1.0000 · Δz/a₀ = -0.7071", lines[-1].text)

    def test_cross_grain_readout_tags_each_grain_and_adds_displayed_components(self):
        from dichromatic_map.vectors import vector_readout_lines
        strained = rotation(0.5) @ np.array([[1.02, 0.01], [0.01, 0.99]])
        measured = self.measure(second_grain=1, deformations=(np.eye(2), strained))
        self.assertTrue(measured.cross_grain and measured.strained)
        self.assertEqual([grain for grain, _ in measured.full], [0, 1])
        lines = vector_readout_lines(measured, display_rotation_deg=90.0)
        self.assertEqual([line.grain for line in lines], [None, 0, 0, 1, 1, 0, 0, 1, 1, None, None])
        self.assertEqual(lines[0].text, "Cross-grain P1→P2 · G1-A → G2-B · axial +0")
        self.assertTrue(lines[3].text.startswith("G2 current (polar) · full: "))
        self.assertTrue(lines[5].text.startswith("G1 lattice [uvw] · full: "))
        self.assertTrue(lines[6].text.startswith("G1 lattice [uvw] · in-plane: "))
        self.assertTrue(lines[8].text.startswith("G2 lattice [uvw] · in-plane: "))
        self.assertEqual(lines[-1].text, "View Δxy/a₀ = (-0.5000, 1.0000)")
        without_view = vector_readout_lines(measured)
        self.assertEqual([line.text for line in without_view], [line.text for line in lines[:-1]])
        # Each grain's in-plane part is the same physical (x, y, 0) vector in its own frame.
        for (_, full), (_, planar) in zip(measured.full, measured.in_plane):
            np.testing.assert_allclose(planar.current_frame @ planar.current,
                                       np.r_[measured.displacement[:2], 0], atol=1e-12)
            np.testing.assert_allclose(planar.lattice_basis @ planar.lattice,
                                       np.r_[measured.displacement[:2], 0], atol=1e-12)
            self.assertGreater(np.linalg.norm(full.current - planar.current), 0.1)

    def test_heterophase_second_grain_uses_its_own_lattice_constant_unit(self):
        from dichromatic_map.vectors import vector_readout_lines
        measured = self.measure(second_grain=1, second_half=(1, 1, 1), second_layer=0,
                                lattice=("FCC", "BCC"), axis=("110", "111"), scale=0.8)
        self.assertTrue(measured.split)
        lines = vector_readout_lines(measured)
        g2 = [line.text for line in lines if line.grain == 1]
        self.assertEqual(len(g2), 2)
        self.assertTrue(g2[0].startswith("G2 current (polar) · full: "))
        self.assertTrue(all("a₀(G2)[" in text for text in g2), g2)
        self.assertTrue(all("a₀(G2)" not in line.text for line in lines if line.grain == 0))
