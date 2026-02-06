##############################################################################
# TripoSG-WebUI — Image-to-3D Web Application
#
# WHAT: A Flask web application that takes a 2D image as input and generates
#       a 3D mesh model using the TripoSG diffusion pipeline. Outputs both
#       GLB (for viewing/sharing) and STL (for 3D printing) formats.
#
# WHY:  TripoSG is a powerful image-to-3D model, but it only ships with a
#       CLI script. This app wraps it in a browser-based UI so anyone can
#       use it without touching a terminal. It also adds mesh solidification
#       for 3D printing, which the original TripoSG does not provide.
#
# HOW:  Flask serves a single-page HTML app with a Three.js 3D viewer.
#       The user uploads an image, the server runs it through background
#       removal (BriaRMBG) and the TripoSG diffusion pipeline to produce
#       a mesh, then exports GLB and STL files for download. Models are
#       loaded on first request and unloaded after each generation to
#       free GPU VRAM.
##############################################################################


# ==========================================================================
# IMPORTS
# ==========================================================================

# gc (garbage collector) — used to force Python to release unreferenced
# objects from memory. We call gc.collect() after GPU operations to ensure
# PyTorch tensors get cleaned up before we ask CUDA to free VRAM.
import gc

# os — file path manipulation and directory creation.
# We use it to build absolute paths to model weights, output directories,
# and static assets relative to wherever this script lives.
import os

# sys — used to modify Python's module search path (sys.path).
# TripoSG's own scripts/ directory isn't a proper Python package, so we
# insert it into sys.path manually so we can import from it.
import sys

# uuid — generates unique identifiers for output files.
# Each generation gets a random 8-character hex ID (e.g., "a3f7b2c1")
# so multiple users/requests don't overwrite each other's files.
import uuid

# tempfile — creates temporary files that we can safely delete later.
# The TripoSG prepare_image() function expects a file path (not a PIL
# Image), so we save the uploaded image to a temp file, process it,
# then delete the temp file.
import tempfile

# torch — PyTorch, the deep learning framework that TripoSG runs on.
# Used for GPU tensor operations, CUDA memory management, and seeding
# the random number generator for reproducible outputs.
import torch

# trimesh — a library for loading, manipulating, and exporting 3D meshes.
# We use it to build Trimesh objects from raw vertex/face arrays, export
# to GLB/STL formats, and run voxelization for mesh solidification.
import trimesh

# numpy — numerical array library. TripoSG outputs vertices and faces
# as arrays. We convert GPU tensors to numpy arrays for trimesh and
# pymeshlab, which don't understand PyTorch tensors.
import numpy as np

# Flask — a lightweight Python web framework.
#   - Flask: the app object that handles HTTP routing
#   - request: access to incoming HTTP request data (form fields, files)
#   - jsonify: converts Python dicts to JSON HTTP responses
#   - send_from_directory: safely serves files from a directory (prevents path traversal)
#   - render_template_string: renders an HTML string as a Jinja2 template
from flask import Flask, request, jsonify, send_from_directory, render_template_string

# PIL (Pillow) — Python Imaging Library for image manipulation.
# We use Image.open() to decode the uploaded image bytes into a PIL Image
# object, then convert it to RGB mode (stripping any alpha channel).
from PIL import Image

# huggingface_hub — downloads pretrained model weights from Hugging Face.
# snapshot_download() fetches an entire model repository (all files) to a
# local directory. On subsequent runs it's a no-op if files already exist.
from huggingface_hub import snapshot_download


# ==========================================================================
# TRIPOSG MODULE IMPORTS
#
# WHAT: Import TripoSG's own pipeline and helper scripts.
#
# WHY:  TripoSG is structured as a research repo, not an installable package.
#       Its scripts/ directory contains the pipeline, image preprocessing,
#       and background removal model — but they aren't on Python's default
#       import path.
#
# HOW:  We insert the scripts/ directory into sys.path[0] (highest priority)
#       so Python can find the triposg package and the helper modules.
#       This avoids modifying any original TripoSG source files.
# ==========================================================================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from scripts.image_process import prepare_image
from scripts.briarmbg import BriaRMBG

# pymeshlab — Python bindings for MeshLab, a mesh processing toolkit.
# We use it for quadric edge collapse decimation, which reduces the number
# of faces in a mesh while preserving shape. This is the "Max faces"
# parameter in the UI. MeshLab's implementation is significantly better
# at preserving surface detail than simple vertex merging.
import pymeshlab


# ==========================================================================
# FLASK APP AND OUTPUT DIRECTORY SETUP
#
# WHAT: Create the Flask app instance and ensure an outputs/ folder exists.
#
# WHY:  Flask needs an app object to register routes. The outputs/ folder
#       stores generated GLB and STL files so they can be served back to
#       the browser for download and 3D viewing.
#
# HOW:  os.makedirs with exist_ok=True creates the directory if missing
#       and does nothing if it already exists (avoids crash on restart).
# ==========================================================================
app = Flask(__name__)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================================
# GLOBAL MODEL REFERENCES
#
# WHAT: Module-level variables that hold the loaded TripoSG pipeline and
#       the BriaRMBG background removal network.
#
# WHY:  Models are large (several GB each) and slow to load. We keep them
#       in globals so load_models() can check if they're already loaded
#       (avoiding redundant loads) and unload_models() can explicitly
#       delete them to free GPU memory after generation.
#
# HOW:  Initialized to None. Set by load_models(), cleared by unload_models().
#       The 'global' keyword in those functions tells Python to modify these
#       module-level variables rather than creating local ones.
# ==========================================================================
pipe = None
rmbg_net = None


# ==========================================================================
# GPU MEMORY MANAGEMENT
#
# WHAT: Two functions that clean up GPU (VRAM) after model inference.
#
# WHY:  The Titan V has 12GB VRAM. TripoSG + BriaRMBG can use most of it.
#       If we don't explicitly free memory, subsequent requests or other
#       GPU programs (like Chrome's WebGL renderer) may fail with OOM.
#       Python's garbage collector doesn't automatically trigger CUDA
#       memory release, so we need to do it manually.
#
# HOW:
#   clear_gpu_memory():
#     1. gc.collect() — tells Python's garbage collector to run immediately,
#        releasing any unreferenced Python objects (including tensor wrappers)
#     2. torch.cuda.empty_cache() — tells PyTorch to release all cached GPU
#        memory blocks back to CUDA. PyTorch normally holds onto freed memory
#        for reuse, but we want it fully released.
#     3. torch.cuda.ipc_collect() — frees shared GPU memory from inter-process
#        communication. Helps when multiple processes share the GPU.
#
#   unload_models():
#     Deletes the global model objects entirely (not just their GPU tensors)
#     and then calls clear_gpu_memory() to flush everything. After this,
#     the models must be re-loaded from disk on the next request.
# ==========================================================================
def clear_gpu_memory():
    """Clear CUDA cache to free GPU memory after generation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def unload_models():
    """Fully unload models from GPU to free all VRAM."""
    global pipe, rmbg_net
    if pipe is not None:
        del pipe
        pipe = None
    if rmbg_net is not None:
        del rmbg_net
        rmbg_net = None
    clear_gpu_memory()


# ==========================================================================
# MODEL LOADING
#
# WHAT: Downloads (if needed) and loads TripoSG + BriaRMBG onto the GPU.
#
# WHY:  Models are ~30GB on disk (TripoSG) and ~200MB (BriaRMBG). First
#       run downloads them from Hugging Face. Subsequent runs load from
#       the local pretrained_weights/ directory. We check if pipe is
#       already loaded to avoid redundant loads within the same server
#       session (though unload_models() resets this after each generation).
#
# HOW:
#   1. snapshot_download() checks if the model files exist locally. If not,
#      it downloads the entire Hugging Face repo. If they do exist, it
#      verifies checksums and skips the download (fast no-op).
#   2. BriaRMBG.from_pretrained() loads the background removal model and
#      .to(device) moves it to GPU. .eval() sets it to inference mode
#      (disables dropout, batch norm training behavior).
#   3. TripoSGPipeline.from_pretrained() loads the main diffusion pipeline.
#      .to(device, dtype) moves it to GPU with float16 precision, which
#      halves VRAM usage vs float32 with negligible quality loss.
# ==========================================================================
def load_models():
    global pipe, rmbg_net
    # Skip loading if models are already in memory
    if pipe is not None:
        return
    device = "cuda"
    # float16 (half precision) uses ~half the VRAM of float32.
    # TripoSG works well in fp16; the quality difference is negligible.
    dtype = torch.float16
    # Build absolute paths to weight directories relative to this script
    triposg_dir = os.path.join(os.path.dirname(__file__), "pretrained_weights", "TripoSG")
    rmbg_dir = os.path.join(os.path.dirname(__file__), "pretrained_weights", "RMBG-1.4")
    # Download model weights from Hugging Face (no-op if already cached)
    snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_dir)
    snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_dir)
    # Load background removal model onto GPU and set to inference mode
    rmbg_net = BriaRMBG.from_pretrained(rmbg_dir).to(device)
    rmbg_net.eval()
    # Load the TripoSG diffusion pipeline onto GPU in float16
    pipe = TripoSGPipeline.from_pretrained(triposg_dir).to(device, dtype)


# ==========================================================================
# MESH SIMPLIFICATION (FACE REDUCTION)
#
# WHAT: Reduces the number of triangles in a mesh to a target face count.
#
# WHY:  TripoSG can produce very dense meshes (hundreds of thousands of
#       faces). Dense meshes are slow to render in the browser's Three.js
#       viewer, slow to transfer over the network, and may be too complex
#       for 3D printing slicers. The "Max faces" UI parameter controls this.
#
# HOW:  Uses PyMeshLab's quadric edge collapse decimation algorithm.
#       This iteratively removes edges from the mesh, choosing edges whose
#       removal causes the least geometric error (measured by quadric error
#       metrics). The result preserves the overall shape much better than
#       naive approaches like random edge removal.
#
#   1. Check if simplification is needed (skip if already under target)
#   2. Create a MeshSet and load the trimesh vertices/faces into it
#   3. Merge very close vertices first (cleans up numerical duplicates)
#   4. Run quadric edge collapse to the target face count
#   5. Extract the result and wrap it back in a trimesh.Trimesh object
# ==========================================================================
def simplify_mesh(mesh, n_faces):
    if mesh.faces.shape[0] > n_faces:
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
        # Merge vertices that are very close together (numerical noise cleanup)
        ms.meshing_merge_close_vertices()
        # Quadric edge collapse: iteratively removes least-important edges
        # until we reach the target face count
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces)
        m = ms.current_mesh()
        return trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix())
    return mesh


# ==========================================================================
# MESH SOLIDIFICATION FOR 3D PRINTING
#
# WHAT: Converts a thin surface mesh into a solid, watertight mesh that
#       3D printing slicers can process.
#
# WHY:  TripoSG outputs surface meshes with zero thickness — they're just
#       a shell of triangles with no interior volume. 3D printing slicers
#       (like OrcaSlicer, PrusaSlicer) need watertight solid geometry to
#       compute toolpaths. Without solidification, the STL files are
#       unprintable — slicers either reject them or produce garbage output.
#
# HOW:  A three-step voxelization pipeline:
#
#   1. mesh.voxelized(pitch) — converts the surface mesh into a 3D grid
#      of voxels (tiny cubes). The pitch parameter controls voxel size in
#      mesh units. Smaller pitch = more voxels = more detail but slower.
#      At pitch=0.01, a unit-sized model gets ~100 voxels per axis.
#
#   2. voxel.fill() — performs a flood-fill operation on the voxel grid.
#      Starting from the outside, it marks all interior voxels as "filled."
#      This is what gives the mesh actual volume/thickness.
#
#   3. voxel.marching_cubes — extracts a smooth triangle mesh from the
#      filled voxel grid using the Marching Cubes algorithm. This classic
#      algorithm walks through the voxel grid and generates triangles at
#      the boundary between filled and empty voxels.
#
#   4. Optional Laplacian smoothing — averages each vertex position with
#      its neighbors to reduce the blocky/staircase artifacts from
#      voxelization. More iterations = smoother but may lose fine detail.
#
# Args:
#   mesh: trimesh.Trimesh input (can be zero-thickness surface)
#   pitch: voxel size in mesh units (default 0.01, range 0.002-0.05)
#   smooth: number of Laplacian smoothing passes (0 = skip smoothing)
#
# Returns:
#   A new trimesh.Trimesh that is watertight and solid — ready for slicing.
# ==========================================================================
def solidify_mesh(mesh, pitch=0.01, smooth=0):
    """Convert surface mesh to solid via voxelization for 3D printing.

    Args:
        mesh: trimesh.Trimesh input (can be zero-thickness surface)
        pitch: voxel size in mesh units (smaller = more detail, slower)
        smooth: number of Laplacian smoothing passes (0 = none)

    Returns:
        Watertight solid trimesh suitable for slicing
    """
    # Step 1: Convert triangle mesh to voxel grid
    voxel = mesh.voxelized(pitch=pitch)
    # Step 2: Flood-fill interior voxels to create solid volume
    voxel = voxel.fill()
    # Step 3: Extract smooth surface mesh from filled voxel grid
    solid = voxel.marching_cubes
    # Step 4 (optional): Smooth away voxelization staircase artifacts
    if smooth > 0:
        trimesh.smoothing.filter_laplacian(solid, iterations=smooth)
    return solid


# ==========================================================================
# 3D MESH GENERATION (CORE INFERENCE PIPELINE)
#
# WHAT: Takes a PIL Image and runs it through the full TripoSG pipeline
#       to produce a 3D triangle mesh.
#
# WHY:  This is the main inference function that ties together model loading,
#       image preprocessing, diffusion inference, and mesh construction.
#       It's separated from the Flask route so it can be tested/called
#       independently of the web server.
#
# HOW:  The pipeline has these stages:
#
#   1. load_models() — ensures TripoSG and BriaRMBG are on the GPU
#
#   2. Save PIL image to a temp file — prepare_image() expects a file
#      path, not a PIL Image object. This is a quirk of TripoSG's original
#      code that we work around rather than modifying their source.
#
#   3. prepare_image() — TripoSG's preprocessing function. It:
#      a. Loads the image from the file path
#      b. Runs BriaRMBG to remove the background (isolate the subject)
#      c. Places the subject on a white background (bg_color=[1,1,1])
#      d. Centers and normalizes the image for the diffusion model
#
#   4. pipe() — runs the TripoSG diffusion pipeline:
#      - image: the preprocessed image tensor
#      - generator: a seeded random number generator for reproducibility.
#        Same seed + same image = same 3D output every time.
#      - num_inference_steps: number of denoising steps. More steps = higher
#        quality but slower. Default 50 is a good balance.
#      - guidance_scale: how closely the output follows the input image.
#        Higher = more faithful to the image but less creative.
#      The output is a list of [vertices, faces] arrays.
#
#   5. Convert GPU tensors to numpy — trimesh and pymeshlab work with numpy
#      arrays, not PyTorch tensors. We immediately move data to CPU and
#      convert to numpy, then delete the GPU tensors to start freeing VRAM.
#
#   6. Build trimesh.Trimesh — construct the mesh object from vertices and
#      faces. Optionally simplify it if max faces was specified.
#
# DECORATOR: @torch.no_grad() disables PyTorch's gradient tracking for this
# entire function. Gradient tracking is only needed for training (backprop).
# Disabling it during inference saves significant memory (~2x) and is faster
# because PyTorch doesn't need to record operations for the backward pass.
# ==========================================================================
@torch.no_grad()
def generate_mesh(image_pil, seed=42, steps=50, guidance=7.0, faces=-1):
    load_models()
    # prepare_image expects a file path, so save PIL image to temp file.
    # We use NamedTemporaryFile to get a unique path, save the image there,
    # pass the path to prepare_image, then delete the temp file.
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image_pil.save(tmp.name)
    # Run TripoSG's image preprocessing: background removal + normalization
    img = prepare_image(tmp.name, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
    os.unlink(tmp.name)  # Clean up temp file immediately
    # Run the diffusion pipeline to generate a 3D mesh from the image
    outputs = pipe(
        image=img,
        # Seeded generator ensures reproducibility: same seed = same output
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=steps,
        guidance_scale=guidance,
    ).samples[0]
    # Convert GPU tensors to numpy arrays immediately, then delete GPU data.
    # This is critical for VRAM management on 12GB cards — holding both the
    # model weights and the output tensors simultaneously can cause OOM.
    verts = outputs[0].cpu().numpy().astype(np.float32) if torch.is_tensor(outputs[0]) else outputs[0].astype(np.float32)
    faces_arr = outputs[1].cpu().numpy() if torch.is_tensor(outputs[1]) else np.ascontiguousarray(outputs[1])
    del outputs, img  # Free GPU memory held by these tensors
    clear_gpu_memory()
    # Build the trimesh object from raw vertex/face arrays
    mesh = trimesh.Trimesh(verts, faces_arr)
    # Optionally reduce face count (faces=-1 means no simplification)
    if faces > 0:
        mesh = simplify_mesh(mesh, faces)
    return mesh


# ==========================================================================
# HTML FRONTEND (SINGLE-PAGE APPLICATION)
#
# WHAT: The entire web UI is embedded as a Python string. Flask serves it
#       directly — no separate HTML files, no build step, no npm.
#
# WHY:  Keeping everything in one file makes deployment trivial. You just
#       need app.py and the TripoSG model code. No frontend toolchain
#       (webpack, vite, etc.) needed. The UI is simple enough that a single
#       HTML page with inline CSS and JS is perfectly adequate.
#
# HOW:  The HTML contains three main parts:
#
#   1. CSS STYLES — Dark theme UI with flexbox layout. Two panels: input
#      (left) and 3D viewer (right). Responsive via flex-wrap.
#
#   2. THREE.JS 3D VIEWER (ES module script) — Initializes a WebGL scene
#      with orbit controls, lighting, and a GLB loader. When a model is
#      generated, loadGLB() fetches the file, centers/scales the model,
#      applies a standard material, and adds it to the scene.
#
#   3. UI LOGIC (regular script) — Handles file drag-and-drop, image
#      preview, form submission via fetch() POST to /generate, and
#      displaying download links when generation completes.
#
# STRUCTURE:
#   - Import map at top tells the browser where to find Three.js modules
#     from the CDN (no local install needed)
#   - Two collapsible sections: "Generation Options" and "Solidify Options"
#   - Generation params: Steps, Guidance, Seed, Max faces
#   - Solidify params: Enabled toggle, Pitch, Smooth
#   - The /generate endpoint returns JSON with GLB/STL download URLs
# ==========================================================================
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TripoSG — Image to 3D</title>
<!-- Import map: tells the browser where to load Three.js ES modules from.
     This avoids needing npm/node_modules — the browser fetches directly
     from the jsDelivr CDN at runtime. Pinned to version 0.160.0 for
     stability (newer versions may change APIs). -->
<script type="importmap">
{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}
</script>
<!-- Inline CSS: dark theme, flexbox layout, responsive two-panel design.
     All styles are inline to keep the app self-contained in one file. -->
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#1a1a2e;color:#e0e0e0;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:2rem}
h1{color:#fff}
.title-row{display:flex;align-items:center;justify-content:center;gap:1rem;margin-bottom:1.5rem}
.title-row img{width:100px;height:100px;object-fit:contain}
.container{display:flex;gap:2rem;flex-wrap:wrap;justify-content:center;width:100%;max-width:1200px}
.panel{background:#16213e;border-radius:12px;padding:1.5rem;flex:1;min-width:320px}
.panel h2{margin-bottom:1rem;font-size:1.1rem;color:#a0c4ff}
#drop-zone{border:2px dashed #444;border-radius:8px;padding:3rem 1rem;text-align:center;cursor:pointer;transition:border-color .2s}
#drop-zone:hover,#drop-zone.dragover{border-color:#a0c4ff}
#drop-zone img{max-width:100%;max-height:300px;margin-top:1rem;border-radius:4px}
input[type=file]{display:none}
button{background:#0f3460;color:#fff;border:none;padding:.7rem 1.5rem;border-radius:6px;cursor:pointer;font-size:1rem;margin-top:1rem;transition:background .2s}
button:hover{background:#1a5276}
button:disabled{opacity:.5;cursor:not-allowed}
#viewer{width:100%;height:400px;border-radius:8px;overflow:hidden;background:#0a0a1a}
#status{margin-top:1rem;min-height:1.5rem;color:#aaa}
.downloads{margin-top:1rem;display:flex;gap:1rem}
.downloads a{color:#a0c4ff;text-decoration:none;padding:.4rem .8rem;border:1px solid #a0c4ff;border-radius:4px;transition:background .2s}
.downloads a:hover{background:#0f3460}
.params{display:flex;gap:1rem;flex-wrap:wrap;margin-top:.8rem}
.params label{font-size:.85rem;color:#aaa}
.params input{width:80px;padding:.3rem;border-radius:4px;border:1px solid #333;background:#0a0a1a;color:#fff}
.advanced-toggle{color:#a0c4ff;cursor:pointer;font-size:.9rem;margin-top:1rem;user-select:none}
.advanced-toggle:hover{text-decoration:underline}
.advanced-section{display:none;margin-top:.8rem;padding:.8rem;background:#0f1a30;border-radius:6px}
.advanced-section.open{display:block}
.advanced-section .params{margin-top:0}
.advanced-section label input[type="checkbox"]{margin-right:.4rem}
</style>
</head>
<body>

<!-- ================================================================
     PAGE HEADER
     WHAT: Title bar with logos and app name.
     WHY:  Branding and navigation to the GitHub project page.
     HOW:  Flexbox row with two logo images flanking the title.
           Both logos link to the GitHub profile.
     ================================================================ -->
<div class="title-row">
  <a href="https://github.com/ErikAllanKincaid/"><img src="/assets/3d-printing-logo.png" alt="3D print maker."></a>
  <h1>TripoSG &mdash; Image to 3D</h1>
  <a href="https://github.com/ErikAllanKincaid/"><img src="/assets/eak.png" alt="Donate"></a>
</div>

<div class="container">

  <!-- ==============================================================
       LEFT PANEL: INPUT AND CONTROLS
       WHAT: Image upload area, generation parameters, and action button.
       WHY:  Users need to select an image and optionally tune parameters
             before generating a 3D model.
       HOW:  A drag-and-drop zone (also clickable) triggers a hidden file
             input. Parameters are in collapsible sections to keep the
             default UI clean. The Generate button POSTs to /generate.
       ============================================================== -->
  <div class="panel">
    <h2>Input</h2>

    <!-- Drop zone: accepts image files via drag-and-drop or click.
         The hidden file input is triggered by the onclick handler.
         When a file is selected, showPreview() displays it inline. -->
    <div id="drop-zone" onclick="document.getElementById('file-input').click()">
      Drop an image here or click to upload
    </div>
    <input type="file" id="file-input" accept="image/*">

    <!-- Generation Options: collapsible section with TripoSG parameters.
         - Steps: number of diffusion denoising iterations (more = better quality, slower)
         - Guidance: classifier-free guidance scale (higher = more faithful to image)
         - Seed: random seed for reproducibility (same seed = same output)
         - Max faces: target face count for mesh simplification (-1 = no limit) -->
    <div class="advanced-toggle" onclick="toggleSection('gen-section', 'gen-arrow')"><span id="gen-arrow">&#9654;</span> Generation Options</div>
    <div class="advanced-section" id="gen-section">
      <div class="params">
        <div><label>Steps</label><br><input type="number" id="steps" value="50" min="1" max="200"></div>
        <div><label>Guidance</label><br><input type="number" id="guidance" value="7.0" step="0.5" min="1"></div>
        <div><label>Seed</label><br><input type="number" id="seed" value="42"></div>
        <div><label>Max faces</label><br><input type="number" id="faces" value="-1" min="-1"></div>
      </div>
    </div>

    <!-- Solidify Options: collapsible section for STL post-processing.
         - Enabled: whether to run voxelization solidification on the STL
         - Pitch: voxel size (smaller = more detail, 0.01 default)
         - Smooth: Laplacian smoothing passes to reduce voxel staircase -->
    <div class="advanced-toggle" onclick="toggleSection('solidify-section', 'solidify-arrow')"><span id="solidify-arrow">&#9654;</span> Solidify Options (STL)</div>
    <div class="advanced-section" id="solidify-section">
      <div class="params">
        <div><label><input type="checkbox" id="solidify-enabled" checked>Enabled</label></div>
        <div><label>Pitch</label><br><input type="number" id="solidify-pitch" value="0.01" step="0.001" min="0.002" max="0.05"></div>
        <div><label>Smooth</label><br><input type="number" id="solidify-smooth" value="0" min="0" max="10"></div>
      </div>
    </div>

    <!-- Link to parameter documentation file -->
    <div style="margin-top:.8rem"><a href="/assets/TripoSG-WebUI_Parameters-explained.txt" target="_blank" style="color:#a0c4ff;font-size:.85rem">Parameter Help</a></div>

    <!-- Generate button: disabled until an image is selected.
         Calls generate() which POSTs the image + params to /generate -->
    <button id="gen-btn" disabled onclick="generate()">Generate 3D Model</button>

    <!-- Status text: shows progress messages during generation -->
    <div id="status"></div>

    <!-- Download links: populated after successful generation with
         GLB (3D viewer/sharing) and STL (3D printing) download links -->
    <div class="downloads" id="downloads"></div>
  </div>

  <!-- ==============================================================
       RIGHT PANEL: 3D VIEWER
       WHAT: An interactive WebGL viewport that displays the generated model.
       WHY:  Users need to see and inspect the 3D model before downloading.
             A built-in viewer avoids needing external software.
       HOW:  Three.js renders the scene. OrbitControls lets users rotate,
             zoom, and pan. The model is loaded as GLB and auto-centered.
       ============================================================== -->
  <div class="panel">
    <h2>3D Viewer</h2>
    <div id="viewer"></div>
  </div>
</div>

<!-- ==================================================================
     THREE.JS 3D VIEWER — ES MODULE
     WHAT: Initializes a WebGL scene with camera, lighting, and controls,
           and provides a loadGLB() function to display generated models.
     WHY:  The browser needs a 3D renderer to preview the generated mesh
           without requiring the user to download it first.
     HOW:  Uses Three.js with ES module imports (via the import map above).
           - PerspectiveCamera at position (0, 1, 3) looking at origin
           - OrbitControls for mouse-driven rotation/zoom/pan
           - Three lights: ambient (0.6) + two directional (0.8 and 0.4)
             to illuminate the model from multiple angles
           - GLTFLoader fetches the .glb file, centers and scales the
             model to fit the viewport, applies a uniform material
     ================================================================== -->
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';

// Scene state — held at module scope so loadGLB can access them
let scene, camera, renderer, controls, currentModel;
const viewer = document.getElementById('viewer');

// init(): Set up the Three.js scene, camera, renderer, lights, and controls.
// Called once when the page loads. The renderer is attached to the #viewer div.
function init(){
  scene = new THREE.Scene();
  // 45-degree field of view, aspect ratio matched to viewer div, near/far clip planes
  camera = new THREE.PerspectiveCamera(45, viewer.clientWidth/viewer.clientHeight, 0.01, 100);
  camera.position.set(0, 1, 3);
  // WebGL renderer with antialiasing for smoother edges
  renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(viewer.clientWidth, viewer.clientHeight);
  renderer.setClearColor(0x0a0a1a);  // Match the dark background
  viewer.appendChild(renderer.domElement);
  // OrbitControls: click-drag to rotate, scroll to zoom, right-click to pan.
  // enableDamping adds inertia — the model keeps spinning briefly after release.
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  // Three-light setup: ambient fill + two directional lights from opposite sides.
  // This ensures no part of the model is in complete shadow.
  const amb = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(amb);
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(2, 4, 3);
  scene.add(dir);
  const dir2 = new THREE.DirectionalLight(0xffffff, 0.4);
  dir2.position.set(-2, -1, -2);
  scene.add(dir2);
  // Start the render loop
  animate();
  // Handle window resize: update camera aspect ratio and renderer size
  // so the 3D view doesn't stretch or crop when the browser is resized
  window.addEventListener('resize', ()=>{
    camera.aspect = viewer.clientWidth/viewer.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(viewer.clientWidth, viewer.clientHeight);
  });
}

// animate(): The render loop. Runs every frame (~60fps) via requestAnimationFrame.
// Updates orbit controls (for damping animation) and renders the scene.
function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera)}

// loadGLB(url): Fetches a .glb file and displays it in the viewer.
// Exposed on window so the non-module UI script can call it.
//
// After loading, the model is:
// 1. Centered: the bounding box center is subtracted from position
// 2. Scaled: normalized so the largest dimension is ~2 units (fits the viewport)
// 3. Re-materialed: all meshes get a uniform MeshStandardMaterial because
//    TripoSG doesn't output texture/color data — just geometry.
//    DoubleSide rendering ensures the model is visible from both sides.
window.loadGLB = function(url){
  console.log('Loading GLB:', url);
  // Remove previously loaded model (if any) before loading the new one
  if(currentModel) scene.remove(currentModel);
  new GLTFLoader().load(url,
    gltf=>{
      console.log('GLB loaded successfully');
      currentModel = gltf.scene;
      // Calculate bounding box to center and scale the model
      const box = new THREE.Box3().setFromObject(currentModel);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3()).length();
      console.log('Model size:', size);
      // Center the model at the origin
      currentModel.position.sub(center);
      // Scale so the model fits nicely in the viewport (2 units across)
      const scale = 2.0 / size;
      currentModel.scale.setScalar(scale);
      // Apply a uniform material to all meshes in the model
      currentModel.traverse(c=>{
        if(c.isMesh){
          c.material = new THREE.MeshStandardMaterial({color:0x8899aa, roughness:0.5, metalness:0.1});
          c.material.side = THREE.DoubleSide;  // Render both front and back faces
        }
      });
      scene.add(currentModel);
      // Reset camera and controls to default position for the new model
      camera.position.set(0, 1, 3);
      controls.reset();
    },
    // Progress callback: logs download percentage
    progress => console.log('Loading progress:', (progress.loaded/progress.total*100).toFixed(1) + '%'),
    // Error callback: logs any loading failures
    error => console.error('GLB load error:', error)
  );
};

init();
</script>

<!-- ==================================================================
     UI LOGIC — FILE UPLOAD, DRAG-AND-DROP, AND FORM SUBMISSION
     WHAT: Handles user interactions: image selection, parameter collection,
           and submitting the generation request to the server.
     WHY:  The Three.js viewer above is an ES module (can't easily share
           scope). This regular script handles all the DOM interactions
           and API calls.
     HOW:
       - File selection via <input> or drag-and-drop both call showPreview()
       - showPreview() reads the file as a data URL and displays it inline
       - generate() collects all form values into a FormData object and
         POSTs it to /generate via the Fetch API
       - On success, download links are shown and loadGLB() is called
       - On failure, the error message is displayed in the status area
     ================================================================== -->
<script>
// Get references to key DOM elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const genBtn = document.getElementById('gen-btn');
const status = document.getElementById('status');
const downloads = document.getElementById('downloads');
let selectedFile = null;  // Holds the currently selected image file

// showPreview(file): Displays a preview of the selected image in the drop zone.
// Uses FileReader to convert the file to a data URL (base64-encoded image)
// that can be displayed in an <img> tag without uploading to the server.
function showPreview(file){
  selectedFile = file;
  genBtn.disabled = false;  // Enable the generate button now that we have an image
  const reader = new FileReader();
  reader.onload = e => {
    // Create or reuse an <img> element inside the drop zone
    let img = dropZone.querySelector('img');
    if(!img){img = document.createElement('img'); dropZone.appendChild(img);}
    img.src = e.target.result;  // Set the image source to the data URL
    dropZone.childNodes[0].textContent = file.name;  // Show filename
  };
  reader.readAsDataURL(file);  // Triggers the onload callback above
}

// Listen for file selection via the hidden <input type="file">
fileInput.addEventListener('change', e => {if(e.target.files[0]) showPreview(e.target.files[0])});

// toggleSection(): Shows/hides a collapsible parameter section and updates
// the arrow icon. Used for both "Generation Options" and "Solidify Options".
function toggleSection(sectionId, arrowId){
  const section = document.getElementById(sectionId);
  const arrow = document.getElementById(arrowId);
  section.classList.toggle('open');
  // Unicode arrows: down-pointing (open) or right-pointing (closed)
  arrow.innerHTML = section.classList.contains('open') ? '&#9660;' : '&#9654;';
}
window.toggleSection = toggleSection;  // Expose to inline onclick handlers

// Drag-and-drop event handlers for the drop zone.
// dragover: prevent default (which would open the file in the browser)
//           and add visual feedback class
// dragleave: remove the visual feedback
// drop: prevent default, remove feedback, and process the dropped file
dropZone.addEventListener('dragover', e => {e.preventDefault(); dropZone.classList.add('dragover')});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('dragover');
  if(e.dataTransfer.files[0]) showPreview(e.dataTransfer.files[0]);
});

// generate(): Collects all parameters and submits them to the /generate endpoint.
// Uses the Fetch API to POST a FormData object (multipart/form-data) containing
// the image file and all parameter values. The server responds with JSON containing
// download URLs for the GLB and STL files.
function generate(){
  if(!selectedFile) return;
  genBtn.disabled = true;  // Prevent double-clicks during generation
  status.textContent = 'Generating 3D model... this may take a few minutes on first run (downloading weights).';
  downloads.innerHTML = '';  // Clear previous download links
  // Build a FormData object with the image and all parameters.
  // FormData automatically sets the Content-Type to multipart/form-data,
  // which is needed for file uploads.
  const fd = new FormData();
  fd.append('image', selectedFile);
  fd.append('steps', document.getElementById('steps').value);
  fd.append('guidance', document.getElementById('guidance').value);
  fd.append('seed', document.getElementById('seed').value);
  fd.append('faces', document.getElementById('faces').value);
  fd.append('solidify_enabled', document.getElementById('solidify-enabled').checked);
  fd.append('solidify_pitch', document.getElementById('solidify-pitch').value);
  fd.append('solidify_smooth', document.getElementById('solidify-smooth').value);
  // POST to /generate and handle the JSON response
  fetch('/generate', {method:'POST', body:fd})
    .then(r => {
      console.log('Response status:', r.status);
      return r.json();
    })
    .then(data => {
      console.log('Response data:', data);
      if(data.error){status.textContent = 'Error: '+data.error; genBtn.disabled=false; return;}
      status.textContent = 'Done!';
      genBtn.disabled = false;
      // Show download links for both GLB and STL formats
      downloads.innerHTML = `<a href="${data.glb}" download>Download GLB</a><a href="${data.stl}" download>Download STL</a>`;
      // Load the GLB into the Three.js viewer for immediate preview
      window.loadGLB(data.glb);
    })
    .catch(err => {console.error('Fetch error:', err); status.textContent = 'Error: '+err; genBtn.disabled=false;});
}
window.generate = generate;  // Expose to the inline onclick handler on the button
</script>
</body>
</html>
"""


# ==========================================================================
# FLASK ROUTES
#
# The app has four routes:
#   GET  /                  — serves the HTML frontend
#   POST /generate          — accepts image upload, runs inference, returns URLs
#   GET  /outputs/<file>    — serves generated GLB/STL files for download
#   GET  /assets/<file>     — serves static assets (logos, docs)
# ==========================================================================


# --------------------------------------------------------------------------
# ROUTE: GET /
# WHAT: Serves the main page (the HTML defined above).
# WHY:  This is the entry point — what users see when they open the app.
# HOW:  render_template_string() processes the HTML string through Jinja2's
#       template engine (though we don't use any Jinja2 features here) and
#       returns it as an HTTP response with Content-Type text/html.
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template_string(HTML)


# --------------------------------------------------------------------------
# ROUTE: POST /generate
# WHAT: Accepts an uploaded image, runs the TripoSG pipeline, and returns
#       download URLs for the generated GLB and STL files.
# WHY:  This is the main API endpoint — it bridges the browser UI and the
#       GPU inference pipeline.
# HOW:
#   1. Extract the uploaded image file from the multipart form data
#   2. Open it as a PIL Image and convert to RGB (strips alpha channel)
#   3. Read all generation parameters from the form data with defaults
#   4. Call generate_mesh() to run the TripoSG diffusion pipeline
#   5. Generate a unique job ID (8 hex chars) for the output filenames
#   6. Export the mesh as GLB (for the 3D viewer and general use)
#   7. If solidification is enabled, voxelize the mesh before exporting STL.
#      If solidification fails (e.g., mesh too complex), fall back to the
#      raw mesh for STL. This ensures the user always gets an STL file.
#   8. Unload models to free GPU VRAM for other processes
#   9. Return JSON with the download URLs: {glb: "/outputs/abc123.glb", ...}
#
# ERROR HANDLING: The entire route is wrapped in try/except. On any error,
# models are unloaded (to prevent VRAM leaks) and the error message is
# returned as JSON with HTTP 500. The traceback is printed server-side
# for debugging.
# --------------------------------------------------------------------------
@app.route("/generate", methods=["POST"])
def generate_route():
    try:
        # Extract image file from the multipart POST request
        f = request.files.get("image")
        if not f:
            return jsonify(error="No image uploaded"), 400
        # Open as PIL Image, convert to RGB (some PNGs have alpha channels
        # that would confuse the background removal model)
        img = Image.open(f.stream).convert("RGB")
        # Read generation parameters from form data, with sensible defaults
        steps = int(request.form.get("steps", 50))
        guidance = float(request.form.get("guidance", 7.0))
        seed = int(request.form.get("seed", 42))
        faces = int(request.form.get("faces", -1))
        # Read solidification parameters
        solidify_enabled = request.form.get("solidify_enabled", "true") == "true"
        solidify_pitch = float(request.form.get("solidify_pitch", 0.01))
        solidify_smooth = int(request.form.get("solidify_smooth", 0))

        # Run the core inference pipeline: image -> 3D mesh
        mesh = generate_mesh(img, seed=seed, steps=steps, guidance=guidance, faces=faces)

        # Generate unique filenames for this job's outputs
        job_id = uuid.uuid4().hex[:8]
        glb_path = os.path.join(OUTPUT_DIR, f"{job_id}.glb")
        stl_path = os.path.join(OUTPUT_DIR, f"{job_id}.stl")
        # Export GLB (used by the Three.js viewer and for sharing)
        mesh.export(glb_path)

        # Export STL — optionally solidified for 3D printing
        if solidify_enabled:
            try:
                # Voxelize + fill + marching cubes to make it printable
                solid_mesh = solidify_mesh(mesh, pitch=solidify_pitch, smooth=solidify_smooth)
                solid_mesh.export(stl_path)
                print(f"Solidified STL saved: {stl_path} (pitch={solidify_pitch}, smooth={solidify_smooth})")
            except Exception as e:
                # Solidification can fail on very complex meshes. Fall back
                # to the raw mesh so the user still gets a downloadable STL.
                print(f"Solidify failed, using original mesh for STL: {e}")
                mesh.export(stl_path)
        else:
            # User disabled solidification — export the raw surface mesh
            mesh.export(stl_path)
            print(f"STL saved without solidify: {stl_path}")

        # Unload all models to fully release GPU VRAM.
        # This means the next request will re-load from disk, but it ensures
        # the GPU is free for other processes between requests.
        unload_models()
        print(f"Models unloaded, GPU memory released")

        # Return JSON with download URLs that the frontend will use
        return jsonify(glb=f"/outputs/{job_id}.glb", stl=f"/outputs/{job_id}.stl")
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Always unload models on error to prevent VRAM leaks
        unload_models()
        return jsonify(error=str(e)), 500


# --------------------------------------------------------------------------
# ROUTE: GET /outputs/<filename>
# WHAT: Serves generated GLB and STL files from the outputs/ directory.
# WHY:  The frontend needs to download these files for the 3D viewer and
#       the download links. send_from_directory is used instead of a raw
#       file read because it handles path traversal security (prevents
#       requests like /outputs/../../etc/passwd).
# --------------------------------------------------------------------------
@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


# --------------------------------------------------------------------------
# ROUTE: GET /assets/<filename>
# WHAT: Serves static assets — logos, parameter documentation file.
# WHY:  Flask's default /static route conflicted with the TripoSG project
#       directory structure, so we use /assets instead. Serves files from
#       the same directory as app.py (logos are placed next to the script).
# HOW:  APP_DIR is the absolute path to the directory containing this script.
#       send_from_directory safely serves files from that directory.
# --------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(APP_DIR, filename)


# ==========================================================================
# APPLICATION ENTRY POINT
#
# WHAT: Starts the Flask development server when this script is run directly.
# WHY:  "python app.py" (or "uv run python app.py") starts the web server.
#       The __name__ == "__main__" guard prevents the server from starting
#       if this file is imported as a module (e.g., by a WSGI server).
# HOW:  host="0.0.0.0" binds to all network interfaces (not just localhost),
#       allowing access from other machines on the network. Port 5000 is
#       Flask's default. debug=False prevents auto-reload and the interactive
#       debugger in production (auto-reload would re-load GPU models
#       unnecessarily; the debugger is a security risk on a network server).
# ==========================================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
