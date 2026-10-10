import dataclasses
import json
import os
from itertools import chain
from typing import Any

import DracoPy  # ty: ignore[unresolved-import]
import numpy as np
import pygltflib

# Typing
from jaxtyping import Float, Int


@dataclasses.dataclass
class Create:
    """Create a 3DTile from a 3D object file."""

    # TODO add other options

    in_path: str
    out_path: str


@dataclasses.dataclass
class Merge:
    """Merge a set of 3DTiles into a big one."""

    # There's no retiling done
    # TODO finish

    # List of tileset.json file to combine
    tile_paths: list[str]
    # Where the output should be saved
    out_path: str

    # TODO validate that the paths are correct


def create_gltf(
    out_path: str,
    points: Float[np.ndarray, "N 3"],
    triangles: Int[np.ndarray, "M 3"],
    *,
    colors: Int[np.ndarray, "N 3"] | None = None,
    compressed: bool = True,
) -> None:
    """Create a gltf file from a set of points and triangles.

    Args:
        out_path: path where to store the gltf file
        points: array of points (in 3D coordinates)
        triangles: array of triangles (in point index)
        colors: array of color for each point (in [0, 255])
        compressed: use draco library to compress the mesh
    """
    # TODO make triangles optional so we can generate point cloud 3DTile

    # Center the points to avoid issues when converting to float32
    center = np.mean(points, axis=0)
    points = points - center[None, :]

    # Create binary buffer
    if compressed:
        binary = DracoPy.encode(
            points.astype(np.float32),
            triangles.astype(np.uint32),
            colors=colors.astype(np.uint8) if colors is not None else None,
        )
    else:
        points_binary = points.astype(np.float32).flatten().tobytes()
        triangles_binary = triangles.astype(np.uint32).flatten().tobytes()
        binary = triangles_binary + points_binary
        if colors is not None:
            colors_binary = colors.astype(np.uint8).flatten().tobytes()
            binary += colors_binary

    # Create glb file
    extensions = {}
    if compressed:
        extensions["KHR_draco_mesh_compression"] = {
            "bufferView": 0,
            "attributes": {
                "POSITION": 1 if colors is not None else 0,
            },
        }
        if colors is not None:
            extensions["KHR_draco_mesh_compression"]["attributes"]["COLOR_0"] = 0

    accessors = [
        pygltflib.Accessor(
            bufferView=0 if not compressed else None,
            componentType=pygltflib.UNSIGNED_INT,
            count=triangles.size,
            type=pygltflib.SCALAR,
            max=[int(triangles.max())],
            min=[int(triangles.min())],
        ),
        pygltflib.Accessor(
            bufferView=1 if not compressed else None,
            componentType=pygltflib.FLOAT,
            count=len(points),
            type=pygltflib.VEC3,
            max=points.max(axis=0).tolist(),
            min=points.min(axis=0).tolist(),
        ),
    ]

    if colors is not None:
        accessors.append(
            pygltflib.Accessor(
                bufferView=2 if not compressed else None,
                componentType=pygltflib.UNSIGNED_BYTE,
                count=len(points),
                normalized=True,
                type=pygltflib.SCALAR,
                max=[255],
                min=[0],
                # type=pygltflib.VEC4,
                # max=[255, 255, 255, 255],
                # min=[0, 0, 0, 255],
            )
        )

    if compressed:
        bufferViews = [
            pygltflib.BufferView(
                buffer=0,
                byteLength=len(binary),
            ),
        ]
    else:
        bufferViews = [
            pygltflib.BufferView(
                buffer=0,
                byteLength=len(triangles_binary),
                target=pygltflib.ELEMENT_ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(triangles_binary),
                byteLength=len(points_binary),
                target=pygltflib.ARRAY_BUFFER,
            ),
        ]
        if colors is not None:
            bufferViews.append(
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(triangles_binary) + len(points_binary),
                    byteLength=len(colors_binary),
                    target=pygltflib.ARRAY_BUFFER,
                )
            )

    # pygltflib has a typing error
    primitive = pygltflib.Primitive(
        attributes=pygltflib.Attributes(
            POSITION=1,
            COLOR_0=2 if colors is not None else None,  # ty: ignore[invalid-argument-type]
        ),
        indices=0,
        extensions=extensions,
    )

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[
            pygltflib.Node(
                matrix=[1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                name="rootNode",
                children=[1],
            ),
            pygltflib.Node(mesh=0, translation=center.tolist()),
        ],
        meshes=[pygltflib.Mesh(primitives=[primitive])],
        accessors=accessors,
        bufferViews=bufferViews,
        buffers=[pygltflib.Buffer(byteLength=len(binary))],
        extensionsUsed=["KHR_draco_mesh_compression"] if compressed else [],
        extensionsRequired=["KHR_draco_mesh_compression"] if compressed else [],
    )
    gltf.set_binary_blob(binary)
    gltf.save(out_path)


def create_3dtile(
    out_path: str,
    points: Float[np.ndarray, "N 3"] | list[Float[np.ndarray, "? 3"]],
    triangles: Int[np.ndarray, "? 3"] | list[Int[np.ndarray, "? 3"]],
    *,
    colors: Int[np.ndarray, "N 3"] | None | list[Int[np.ndarray, "? 3"] | None] = None,
    geomErrors: float | list[float] = 0.0,
    compressed: bool = True,
) -> None:
    """Create a single tile 3DTile model from the points and triangles provided.

    Args:
        out_path: path where to store the gltf file
        points: array of points (in 3D coordinates), a list corresponds to each scale when doing LOD
        triangles: array of triangles (in point index), a list corresponds to each scale when doing LOD
        colors: array of color for each point (in [0, 255]), a list corresponds to each scale when doing LOD
        geomErrors: the geometric error associated to each scale
        compressed: use draco library to compress the mesh
    """
    # TODO make triangles optional so we can generate point cloud 3DTile -> low priority because 3DTiles viewers (Giro3D) don't seem to support point clouds
    # Need to refont the compression scheme to also support meshopt because draco point compression is not supported by KHR_draco_compression

    # FIXME for the lod we need the map of points (otherwise we need to estimate it again which is not optimal)
    # Handle degenerate cases ?

    # TODO function description where it specifies the order of details in the list (simpler is to fine coarse)

    # Transform input into lists (if necessary)
    if isinstance(points, np.ndarray):
        points = [points]
    if isinstance(triangles, np.ndarray):
        triangles = [triangles]
    if colors is None:
        colors = [None] * len(points)
    elif isinstance(colors, np.ndarray):
        colors = [colors]
    if isinstance(geomErrors, (int, float)):
        geomErrors = [geomErrors]

    assert (
        len(points) == len(triangles)
        and len(points) == len(colors)
        and len(points) == len(geomErrors)
    ), (
        "Invalid parameters. The points list, triangles list, colors list (if provided) and geometric errors list (if provided) should be of the same length."
    )

    mini = np.min(points[0], axis=0)
    maxi = np.max(points[0], axis=0)
    center = (maxi + mini) / 2

    # Create tileset.json file
    tile = {
        "asset": {
            "version": "1.0",
        },
        "geometricError": np.sqrt(
            (maxi[0] - mini[0]) ** 2
            + (maxi[1] - mini[1]) ** 2
            + (maxi[2] - mini[2]) ** 2
        ),
    }

    leaf = None
    for lod, (ps, ts, cs, er) in enumerate(zip(points, triangles, colors, geomErrors)):
        tmp: dict[str, Any] = {
            "boundingVolume": {
                "box": [
                    center[0],
                    center[1],
                    center[2],
                    (maxi[0] - mini[0]) / 2,
                    0,
                    0,
                    0,
                    (maxi[1] - mini[1]) / 2,
                    0,
                    0,
                    0,
                    (maxi[2] - mini[2]) / 2,
                ]
            },
            "geometricError": er,
            "refine": "REPLACE",
            "content": {"uri": f"{len(points) - 1 - lod}.glb"},
        }

        if leaf is not None:
            tmp["children"] = [leaf]

        leaf = tmp

        # FIXME should we stop compressing at some point ?
        create_gltf(
            os.path.join(out_path, f"{len(points) - 1 - lod}.glb"),
            ps,
            ts,
            colors=cs,
            compressed=compressed,
        )

    tile["root"] = leaf

    with open(os.path.join(out_path, "tileset.json"), "w") as f:
        json.dump(tile, f)


# TODO it is short but is it efficient ?
def get_volumes(tile: dict[str, Any]) -> list[dict[int, float]]:
    """Extract a list of bound volumes for the given tile."""
    if "children" in tile:
        return list(
            chain(
                *[get_volumes(child) for child in tile["children"]],
                [tile["boundingVolume"]["box"]],
            )
        )
    return [tile["boundingVolume"]["box"]]


def merge(tile_paths: list[str], out_path: str) -> None:
    """Combine a set of existing tiles.

    Args:
        tile_paths: list of tiles to be merged
        out_path: path where to save the merged tile
    """

    # TODO mention that tile_paths must be relative to the out_path !
    # TODO add a warning because if the tile paths are absolute then the original tiles can't be moved !
    # In addition, serving the new tile can be tricky

    def read_file(tile_path: str):
        with open(os.path.join(out_path, tile_path), "r") as f:
            data = json.load(f)
        return data

    tiles = [read_file(tile_path) for tile_path in tile_paths]

    # Extract relevant corners from all the tiles
    boxes = [
        np.asarray(
            [
                [
                    volume[0] - volume[3],
                    volume[1] - volume[7],
                    volume[2] - volume[11],
                ],
                [
                    volume[0] + volume[3],
                    volume[1] + volume[7],
                    volume[2] + volume[11],
                ],
            ]
        )
        for tile in tiles
        for volume in get_volumes(tile["root"])
    ]
    boxes = np.concatenate(boxes, axis=0)
    mini = np.min(boxes, axis=0)
    maxi = np.max(boxes, axis=0)

    # Create the englobing bounding box
    box = [
        0.0,
        0.0,
        0.0,
        (maxi[0] - mini[0]) / 2,
        0.0,
        0.0,
        0.0,
        (maxi[1] - mini[1]) / 2,
        0.0,
        0.0,
        0.0,
        (maxi[2] - mini[2]) / 2,
    ]
    center = [(maxi[0] + mini[0]) / 2, (maxi[1] + mini[1]) / 2, (maxi[2] + mini[2]) / 2]

    children = [
        {
            "boundingVolume": tile["root"]["boundingVolume"],
            "geometricError": tile["root"]["geometricError"],
            "refine": "REPLACE",
            # Since we reuse the previous tileset (to avoid changing uris), we need to compensate the double shift (one by the original and one by the newly created)
            "transform": [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                -center[0],
                -center[1],
                -center[2],
                1.0,
            ],
            "content": {"uri": tile_path},
        }
        for tile, tile_path in zip(tiles, tile_paths)
    ]

    merged_tile = {
        "asset": {
            "version": "1.0",
        },
        "root": {
            "boundingVolume": {"box": box},
            "transform": [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                center[0],
                center[1],
                center[2],
                1.0,
            ],
            "geometricError": np.sqrt(box[3] ** 2 + box[7] ** 2 + box[11] ** 2),
            "refine": "REPLACE",
            "children": children,
        },
    }

    with open(os.path.join(out_path, "tileset.json"), "w") as f:
        json.dump(merged_tile, f)


def cli():
    import tyro

    cmd = tyro.cli(Create | Merge)

    if isinstance(cmd, Create):
        # create_cli(cmd.in_path, cmd.out_path)
        assert False, "Feature not yet available"
    elif isinstance(cmd, Merge):
        merge(cmd.tile_paths, cmd.out_path)
    else:
        assert False, "This should not be possible"
