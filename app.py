import gc
import os
import sys
import uuid
import tempfile
import torch
import trimesh
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from PIL import Image
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from scripts.image_process import prepare_image
from scripts.briarmbg import BriaRMBG
import pymeshlab

app = Flask(__name__)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global model refs — loaded once
pipe = None
rmbg_net = None


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


def load_models():
    global pipe, rmbg_net
    if pipe is not None:
        return
    device = "cuda"
    dtype = torch.float16
    triposg_dir = os.path.join(os.path.dirname(__file__), "pretrained_weights", "TripoSG")
    rmbg_dir = os.path.join(os.path.dirname(__file__), "pretrained_weights", "RMBG-1.4")
    snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_dir)
    snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_dir)
    rmbg_net = BriaRMBG.from_pretrained(rmbg_dir).to(device)
    rmbg_net.eval()
    pipe = TripoSGPipeline.from_pretrained(triposg_dir).to(device, dtype)


def simplify_mesh(mesh, n_faces):
    if mesh.faces.shape[0] > n_faces:
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces)
        m = ms.current_mesh()
        return trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix())
    return mesh


def solidify_mesh(mesh, pitch=0.01, smooth=0):
    """Convert surface mesh to solid via voxelization for 3D printing.

    Args:
        mesh: trimesh.Trimesh input (can be zero-thickness surface)
        pitch: voxel size in mesh units (smaller = more detail, slower)
        smooth: number of Laplacian smoothing passes (0 = none)

    Returns:
        Watertight solid trimesh suitable for slicing
    """
    voxel = mesh.voxelized(pitch=pitch)
    voxel = voxel.fill()
    solid = voxel.marching_cubes
    if smooth > 0:
        trimesh.smoothing.filter_laplacian(solid, iterations=smooth)
    return solid


@torch.no_grad()
def generate_mesh(image_pil, seed=42, steps=50, guidance=7.0, faces=-1):
    load_models()
    # prepare_image expects a file path, so save PIL image to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image_pil.save(tmp.name)
    img = prepare_image(tmp.name, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
    os.unlink(tmp.name)
    outputs = pipe(
        image=img,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=steps,
        guidance_scale=guidance,
    ).samples[0]
    # Convert to numpy immediately and free GPU tensors
    verts = outputs[0].cpu().numpy().astype(np.float32) if torch.is_tensor(outputs[0]) else outputs[0].astype(np.float32)
    faces_arr = outputs[1].cpu().numpy() if torch.is_tensor(outputs[1]) else np.ascontiguousarray(outputs[1])
    del outputs, img
    clear_gpu_memory()
    mesh = trimesh.Trimesh(verts, faces_arr)
    if faces > 0:
        mesh = simplify_mesh(mesh, faces)
    return mesh


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TripoSG — Image to 3D</title>
<script type="importmap">
{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}
</script>
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
<div class="title-row">
  <a href="https://github.com/ErikAllanKincaid/"><img src="/assets/3d-printing-logo.png" alt="3D print maker."></a>
  <h1>TripoSG &mdash; Image to 3D</h1>
  <a href="https://github.com/ErikAllanKincaid/"><img src="/assets/eak.png" alt="Donate"></a>
</div>
<div class="container">
  <div class="panel">
    <h2>Input</h2>
    <div id="drop-zone" onclick="document.getElementById('file-input').click()">
      Drop an image here or click to upload
    </div>
    <input type="file" id="file-input" accept="image/*">
    <div class="params">
      <div><label>Steps</label><br><input type="number" id="steps" value="50" min="1" max="200"></div>
      <div><label>Guidance</label><br><input type="number" id="guidance" value="7.0" step="0.5" min="1"></div>
      <div><label>Seed</label><br><input type="number" id="seed" value="42"></div>
      <div><label>Max faces</label><br><input type="number" id="faces" value="-1" min="-1"></div>
      <div style="display:flex;align-items:flex-end"><a href="/assets/TripoSG-WebUI_Parameters-explained.txt" target="_blank" style="color:#a0c4ff;font-size:.85rem">Parameter Help</a></div>
    </div>
    <div class="advanced-toggle" onclick="toggleAdvanced()"><span id="adv-arrow">&#9654;</span> Solidify Options (STL)</div>
    <div class="advanced-section" id="advanced-section">
      <div class="params">
        <div><label><input type="checkbox" id="solidify-enabled" checked>Enabled</label></div>
        <div><label>Pitch</label><br><input type="number" id="solidify-pitch" value="0.01" step="0.001" min="0.002" max="0.05"></div>
        <div><label>Smooth</label><br><input type="number" id="solidify-smooth" value="0" min="0" max="10"></div>
      </div>
    </div>
    <button id="gen-btn" disabled onclick="generate()">Generate 3D Model</button>
    <div id="status"></div>
    <div class="downloads" id="downloads"></div>
  </div>
  <div class="panel">
    <h2>3D Viewer</h2>
    <div id="viewer"></div>
  </div>
</div>

<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';

let scene, camera, renderer, controls, currentModel;
const viewer = document.getElementById('viewer');

function init(){
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(45, viewer.clientWidth/viewer.clientHeight, 0.01, 100);
  camera.position.set(0, 1, 3);
  renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(viewer.clientWidth, viewer.clientHeight);
  renderer.setClearColor(0x0a0a1a);
  viewer.appendChild(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  const amb = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(amb);
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(2, 4, 3);
  scene.add(dir);
  const dir2 = new THREE.DirectionalLight(0xffffff, 0.4);
  dir2.position.set(-2, -1, -2);
  scene.add(dir2);
  animate();
  window.addEventListener('resize', ()=>{
    camera.aspect = viewer.clientWidth/viewer.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(viewer.clientWidth, viewer.clientHeight);
  });
}

function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera)}

window.loadGLB = function(url){
  console.log('Loading GLB:', url);
  if(currentModel) scene.remove(currentModel);
  new GLTFLoader().load(url,
    gltf=>{
      console.log('GLB loaded successfully');
      currentModel = gltf.scene;
      const box = new THREE.Box3().setFromObject(currentModel);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3()).length();
      console.log('Model size:', size);
      currentModel.position.sub(center);
      const scale = 2.0 / size;
      currentModel.scale.setScalar(scale);
      currentModel.traverse(c=>{
        if(c.isMesh){
          c.material = new THREE.MeshStandardMaterial({color:0x8899aa, roughness:0.5, metalness:0.1});
          c.material.side = THREE.DoubleSide;
        }
      });
      scene.add(currentModel);
      camera.position.set(0, 1, 3);
      controls.reset();
    },
    progress => console.log('Loading progress:', (progress.loaded/progress.total*100).toFixed(1) + '%'),
    error => console.error('GLB load error:', error)
  );
};

init();
</script>

<script>
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const genBtn = document.getElementById('gen-btn');
const status = document.getElementById('status');
const downloads = document.getElementById('downloads');
let selectedFile = null;

function showPreview(file){
  selectedFile = file;
  genBtn.disabled = false;
  const reader = new FileReader();
  reader.onload = e => {
    let img = dropZone.querySelector('img');
    if(!img){img = document.createElement('img'); dropZone.appendChild(img);}
    img.src = e.target.result;
    dropZone.childNodes[0].textContent = file.name;
  };
  reader.readAsDataURL(file);
}

fileInput.addEventListener('change', e => {if(e.target.files[0]) showPreview(e.target.files[0])});

function toggleAdvanced(){
  const section = document.getElementById('advanced-section');
  const arrow = document.getElementById('adv-arrow');
  section.classList.toggle('open');
  arrow.innerHTML = section.classList.contains('open') ? '&#9660;' : '&#9654;';
}
window.toggleAdvanced = toggleAdvanced;
dropZone.addEventListener('dragover', e => {e.preventDefault(); dropZone.classList.add('dragover')});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('dragover');
  if(e.dataTransfer.files[0]) showPreview(e.dataTransfer.files[0]);
});

function generate(){
  if(!selectedFile) return;
  genBtn.disabled = true;
  status.textContent = 'Generating 3D model... this may take a few minutes on first run (downloading weights).';
  downloads.innerHTML = '';
  const fd = new FormData();
  fd.append('image', selectedFile);
  fd.append('steps', document.getElementById('steps').value);
  fd.append('guidance', document.getElementById('guidance').value);
  fd.append('seed', document.getElementById('seed').value);
  fd.append('faces', document.getElementById('faces').value);
  fd.append('solidify_enabled', document.getElementById('solidify-enabled').checked);
  fd.append('solidify_pitch', document.getElementById('solidify-pitch').value);
  fd.append('solidify_smooth', document.getElementById('solidify-smooth').value);
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
      downloads.innerHTML = `<a href="${data.glb}" download>Download GLB</a><a href="${data.stl}" download>Download STL</a>`;
      window.loadGLB(data.glb);
    })
    .catch(err => {console.error('Fetch error:', err); status.textContent = 'Error: '+err; genBtn.disabled=false;});
}
window.generate = generate;
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/generate", methods=["POST"])
def generate_route():
    try:
        f = request.files.get("image")
        if not f:
            return jsonify(error="No image uploaded"), 400
        img = Image.open(f.stream).convert("RGB")
        steps = int(request.form.get("steps", 50))
        guidance = float(request.form.get("guidance", 7.0))
        seed = int(request.form.get("seed", 42))
        faces = int(request.form.get("faces", -1))
        solidify_enabled = request.form.get("solidify_enabled", "true") == "true"
        solidify_pitch = float(request.form.get("solidify_pitch", 0.01))
        solidify_smooth = int(request.form.get("solidify_smooth", 0))

        mesh = generate_mesh(img, seed=seed, steps=steps, guidance=guidance, faces=faces)

        job_id = uuid.uuid4().hex[:8]
        glb_path = os.path.join(OUTPUT_DIR, f"{job_id}.glb")
        stl_path = os.path.join(OUTPUT_DIR, f"{job_id}.stl")
        mesh.export(glb_path)

        # Solidify mesh for 3D printing (if enabled)
        if solidify_enabled:
            try:
                solid_mesh = solidify_mesh(mesh, pitch=solidify_pitch, smooth=solidify_smooth)
                solid_mesh.export(stl_path)
                print(f"Solidified STL saved: {stl_path} (pitch={solidify_pitch}, smooth={solidify_smooth})")
            except Exception as e:
                print(f"Solidify failed, using original mesh for STL: {e}")
                mesh.export(stl_path)
        else:
            mesh.export(stl_path)
            print(f"STL saved without solidify: {stl_path}")

        # Fully unload models to release GPU memory
        unload_models()
        print(f"Models unloaded, GPU memory released")

        return jsonify(glb=f"/outputs/{job_id}.glb", stl=f"/outputs/{job_id}.stl")
    except Exception as e:
        import traceback
        traceback.print_exc()
        unload_models()
        return jsonify(error=str(e)), 500


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


APP_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(APP_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
