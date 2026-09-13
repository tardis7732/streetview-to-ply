# GPU backend setup

The desktop GUI and the reconstruction backend are separate installations. A
fresh checkout can browse the map and save selections. Generation requires an
operator profile, a Linux CUDA host, installed model weights, and the external
programs below. This repository does not include those weights or binaries.
The published source has not been trained end to end on a fresh GPU host.

## Core reconstruction environment

Use Python 3.11 or 3.12, an NVIDIA driver, a CUDA toolkit/compiler compatible
with the installed PyTorch build, Git, and SSH/SCP. Install CUDA-enabled PyTorch
for the host first, then install the core packages from the repository root:

```bash
python -m venv .venv-gpu
. .venv-gpu/bin/activate
# Install the host-compatible PyTorch build before the next command.
python -m pip install -r requirements-gpu.txt
```

The named reconstruction recipe uses PyCOLMAP 4.2.0 and gsplat 1.5.3. The
requirements file is an installation starting point, not a lock for all GPU
and model combinations. Preserve the working environment with `pip freeze`.
gsplat can compile CUDA code on first use; use its
[official installation instructions](https://github.com/nerfstudio-project/gsplat#installation)
for the compiler and PyTorch combination.

The panorama recipe invokes a separately installed **Brush 0.3** executable.
Obtain or build that version from the
[Brush project](https://github.com/ArthurBrussee/brush), confirm its `--version`,
and configure its absolute Linux path and SHA-256. A newer Brush version is
not an automatic replacement for this recipe.

`brush_refine.renderer_library` must identify the actual compiled gsplat
extension loaded by the training environment. Its SHA-256 is installation
specific. `brush_refine_runtime.py` also checks the 1.5.3 Python renderer and
CUDA wrapper source hashes. This prevents silently mixing a newer renderer
with the saved recipe. Configure the installed extension after building it;
do not copy another machine's compiled library or replace the expected source
hashes merely to suppress a mismatch.

## Image models

Install model runtimes independently when their dependencies conflict. All
model calls use local files. Download the selected model revision through its
official distribution first, then fill in the local paths and hashes described
in [CONFIGURATION.md](CONFIGURATION.md).

| Stage | Required local model/runtime | Operator section |
|---|---|---|
| Object masks | Transformers `Sam3Model` and `Sam3Processor`; SAM3 weights and processor | `panorama_preprocess.sam_segmentation` |
| Instance verification | Transformers `RTDetrV2ForObjectDetection` and `RTDetrImageProcessor`; RT-DETR-v2 weights | `sam_segmentation.instance_verifier` |
| Broad sky mask | Transformers `AutoModelForSemanticSegmentation` and `AutoImageProcessor`; semantic model and its processor | `panorama_preprocess.sky_segmentation` |
| Whole-panorama removal | Diffusers `Flux2KleinPipeline`, its full model snapshot, and the operator's verified model receipt | `panorama_preprocess.flux` |
| Cleanup depth | Two pinned DA3 source trees and model snapshots: metric/nested and known-pose branches | `sky_depth.metric`, `sky_depth.pose` |

The connected masks use SAM3 plus a separate detector; changing only a model
name is insufficient because the adapter also checks files, labels, and
processor identity. See the official
[SAM3 adapter](https://huggingface.co/docs/transformers/main/en/model_doc/sam3)
and [FLUX.2 pipeline](https://huggingface.co/docs/diffusers/main/api/pipelines/flux2)
documentation for their installation requirements. Model source and weight
licenses remain those of their respective distributions.

The FLUX model receipt is JSON containing `status: "downloaded_verified"`,
`revision`, an absolute `model_path`, and a `files` array. Every array row has
`path` relative to the snapshot, `bytes`, and `sha256`. Inventory the complete
downloaded snapshot after validating its intended revision. The operator
configuration binds the receipt itself with `model_receipt_sha256`; a receipt
does not download or authorize a model. The adapter uses complete 2:1 ERP input
and mask-only native-grid composition; it does not expose an external image
generation service.

For DA3, retain each checkout's Git metadata. `da3_raw_depth.py` verifies
`repo_revision`, a clean tracked `src/depth_anything_3` tree,
`model.safetensors`, and `config.json` before inference. Its `side` is fixed at
504 for the connected recipe. Use the
[DA3 installation documentation](https://github.com/ByteDance-Seed/Depth-Anything-3)
for the selected revisions and record the resulting environment. Raw DA3 depth
is an abstention aid for cleanup; it is not calibrated ground truth. Disabling
the GUI depth-cleanup option skips this inference and its associated cleanup.

The generic multi-view backend retains a separately configured depth-only
UniSHARP/UniK3D auxiliary adapter in `depth_prior.py`. It is inactive in the
named panorama/Brush recipe. It never exports predicted Gaussian parameters,
and it is not a single-panorama generation feature. Installing it is optional
unless an operator explicitly enables that generic depth-prior path.

## SSH worker

Choose a configured SSH alias and a private remote workspace. The profile
needs three absolute Linux paths: a Python launcher, a bootstrap worker file,
and a jobs directory. Copy `tools/streetview_engine/remote_worker.py` to the
configured worker location. The local GUI uploads the required product source
bundle into each new job; it does not install Python packages or model weights.

An operator-owned launcher can be as small as:

```sh
#!/bin/sh
set -eu
exec /absolute/path/to/.venv-gpu/bin/python "$@"
```

Make it executable and verify that the SSH alias can run it. Configure local
settings and backend paths using [CONFIGURATION.md](CONFIGURATION.md).
The service uses Linux process groups and renewable leases; keep the GUI
server running while it owns a remote job. A settings or source change
invalidates saved execution snapshots: register a fresh preset instead of
rewriting an old job's hashes.

## Unreal preview

PLY export is independent of Unreal. To enable **Open in Unreal**, separately
install Unreal Editor and MLSLabsRenderer, and configure an existing project
with `MLSLabsRenderer`, `PythonScriptPlugin`, and `EditorScriptingUtilities`
enabled. The adapter requires an ASCII project path on Windows and creates a
new preview level. The repository contains only its authored Python template;
it does not redistribute Unreal, the plugin, or a project containing them.
