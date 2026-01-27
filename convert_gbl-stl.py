import trimesh
import sys

if len(sys.argv) > 2:
    glb_file = sys.argv[1]
    stl_file = sys.argv[2]
    mesh = trimesh.load_mesh(glb_file)
    mesh.export(stl_file)
    print(f"Converted {glb_file} to {stl_file}")
else:
    print("Usage: python convert.py <input.glb> <output.stl>")
