"""Orientation regressions; run with the optional trimesh conversion dependency."""

import io
import unittest

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

from scripts.convert_glb_to_obj import orient_mesh_outward


@unittest.skipIf(trimesh is None, "trimesh is available in the py313 conversion environment")
class MeshOrientationTests(unittest.TestCase):
    def cube(self, *, hard_edges=False):
        box = trimesh.creation.box()
        if not hard_edges:
            return box
        return trimesh.Trimesh(
            vertices=box.triangles.reshape(-1, 3),
            faces=np.arange(len(box.faces) * 3).reshape(-1, 3),
            vertex_normals=np.repeat(box.face_normals, 3, axis=0),
            process=False,
        )

    def reverse_faces(self, mesh, indices):
        """Make an input whose normals follow the selected reversed faces."""
        normals = mesh.vertex_normals.copy()
        faces = mesh.faces.copy()
        faces[indices] = faces[indices, ::-1]
        mesh.faces = faces
        normals[np.unique(faces[indices])] *= -1
        mesh.vertex_normals = normals

    def welded(self, mesh):
        vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
        return trimesh.Trimesh(vertices, inverse[mesh.faces], process=False)

    def assert_outward(self, mesh):
        welded = self.welded(mesh)
        self.assertTrue(welded.is_winding_consistent)
        for component in welded.split(only_watertight=False):
            self.assertGreater(component.volume, 0)

    def test_hard_edge_duplicates_keep_positions_and_normals(self):
        mesh = self.cube(hard_edges=True)
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
        normals = mesh.vertex_normals.copy()
        mesh.metadata["name"] = "white_shell"
        mesh.visual.face_colors = np.tile([240, 240, 240, 255], (len(faces), 1))
        colors = mesh.visual.face_colors.copy()
        self.reverse_faces(mesh, np.array([0, 3, 5, 9]))

        self.assertEqual(orient_mesh_outward(mesh), 4)

        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(np.sort(mesh.faces, axis=1), np.sort(faces, axis=1))
        np.testing.assert_allclose(mesh.vertex_normals, normals, atol=1e-14)
        np.testing.assert_array_equal(mesh.visual.face_colors, colors)
        self.assertEqual(mesh.metadata["name"], "white_shell")
        self.assertEqual(len(mesh.vertices), 36)
        self.assert_outward(mesh)

    def test_mixed_winding_recomputes_conflicting_shared_normals(self):
        mesh = self.cube()
        vertices = mesh.vertices.copy()
        self.reverse_faces(mesh, np.array([1, 4, 6, 10]))

        self.assertEqual(orient_mesh_outward(mesh), 4)

        expected = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False).vertex_normals
        np.testing.assert_allclose(mesh.vertex_normals, expected, atol=1e-14)
        np.testing.assert_array_equal(mesh.vertices, vertices)
        self.assert_outward(mesh)

    def test_inward_mesh_flips_all_normals(self):
        mesh = self.cube(hard_edges=True)
        normals = mesh.vertex_normals.copy()
        self.reverse_faces(mesh, np.arange(len(mesh.faces)))

        self.assertEqual(orient_mesh_outward(mesh), 12)

        np.testing.assert_array_equal(mesh.vertex_normals, normals)
        self.assert_outward(mesh)

    def test_components_use_independent_centers_and_volume_signs(self):
        for offset in ([0, 0, 0], [1e5, -2e5, 3e5]):
            with self.subTest(offset=offset):
                first = self.cube(hard_edges=True)
                second = self.cube(hard_edges=True)
                second.apply_translation([4, 2, -3])
                self.reverse_faces(second, np.arange(len(second.faces)))
                mesh = trimesh.util.concatenate([first, second])
                mesh.apply_translation(offset)
                vertices = mesh.vertices.copy()

                self.assertEqual(orient_mesh_outward(mesh), 12)

                np.testing.assert_array_equal(mesh.vertices, vertices)
                self.assert_outward(mesh)

    def test_planar_component_retains_original_winding_and_normals(self):
        for offset in ([0, 0, 0], [5, 7, 11]):
            with self.subTest(offset=offset):
                # Deliberately mixed winding on an offset plane; no volume sign.
                mesh = trimesh.Trimesh(
                    vertices=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]) + offset,
                    faces=[[0, 1, 2], [0, 3, 2]], process=False,
                )
                faces = mesh.faces.copy()
                normals = mesh.vertex_normals.copy()

                self.assertEqual(orient_mesh_outward(mesh), 0)

                np.testing.assert_array_equal(mesh.faces, faces)
                np.testing.assert_array_equal(mesh.vertex_normals, normals)

    def test_near_zero_volume_patch_retains_direction_at_any_scale(self):
        for scale in (1e-3, 1, 1e3):
            with self.subTest(scale=scale):
                mesh = trimesh.Trimesh(
                    vertices=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1e-9], [0, 1, 0]]) * scale,
                    faces=[[0, 2, 1], [0, 3, 2]], process=False,
                )
                faces = mesh.faces.copy()
                normals = mesh.vertex_normals.copy()

                self.assertEqual(orient_mesh_outward(mesh), 0)

                np.testing.assert_array_equal(mesh.faces, faces)
                np.testing.assert_array_equal(mesh.vertex_normals, normals)

    def test_planar_and_inward_components_are_handled_separately(self):
        box = self.cube(hard_edges=True)
        self.reverse_faces(box, np.arange(len(box.faces)))
        plane = trimesh.Trimesh(
            vertices=[[5, 0, 2], [6, 0, 2], [6, 1, 2], [5, 1, 2]],
            faces=[[0, 2, 1], [0, 3, 2]], process=False,
        )
        mesh = trimesh.util.concatenate([box, plane])
        planar_faces = mesh.faces[-2:].copy()
        planar_normals = mesh.vertex_normals[-4:].copy()

        self.assertEqual(orient_mesh_outward(mesh), 12)

        np.testing.assert_array_equal(mesh.faces[-2:], planar_faces)
        np.testing.assert_array_equal(mesh.vertex_normals[-4:], planar_normals)

    def test_outward_mesh_keeps_custom_smooth_normals(self):
        mesh = self.cube(hard_edges=True)
        normals = mesh.vertex_normals.copy()
        normals[:, 0] += 0.1
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        mesh.vertex_normals = normals
        faces = mesh.faces.copy()

        self.assertEqual(orient_mesh_outward(mesh), 0)

        np.testing.assert_array_equal(mesh.faces, faces)
        np.testing.assert_array_equal(mesh.vertex_normals, normals)

    def test_repair_and_obj_export_are_repeatable(self):
        mesh = self.cube(hard_edges=True)
        self.reverse_faces(mesh, np.array([0, 2, 5, 8, 11]))
        self.assertEqual(orient_mesh_outward(mesh), 5)
        first_export = mesh.export(file_type="obj", include_normals=True)
        normals = mesh.vertex_normals.copy()

        self.assertEqual(orient_mesh_outward(mesh), 0)
        self.assertEqual(mesh.export(file_type="obj", include_normals=True), first_export)
        np.testing.assert_array_equal(mesh.vertex_normals, normals)
        reloaded = trimesh.load_mesh(
            io.StringIO(first_export), file_type="obj", process=False, maintain_order=True,
        )
        self.assertEqual(orient_mesh_outward(reloaded), 0)
        self.assert_outward(reloaded)
        np.testing.assert_allclose(reloaded.vertex_normals, normals, atol=1e-8)

    def test_empty_mesh_is_unchanged(self):
        self.assertEqual(orient_mesh_outward(trimesh.Trimesh()), 0)


if __name__ == "__main__":
    unittest.main()
