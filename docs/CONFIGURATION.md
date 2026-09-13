# Configuration

The GUI can browse the map, preview streetview, select captures, and save a selection before a GPU backend is configured. Generating a PLY is an explicit action. Opening the GUI or registering a preset does not start collection or training.

Follow [Local GPU setup](GPU_SETUP.md) to prepare models and tools on the same Linux / WSL2 computer. The default backend runs local Python stages.

## Start the GUI

From an installed checkout:

```sh
python -m tools.streetview_app.server --data-dir tools/streetview_app/data --open
```

The server listens on `127.0.0.1:8765`. This is also the data directory used by the default launcher and Windows shortcut. Use the same `--data-dir` in the configuration command below. A fresh installation reports generation as unconfigured; this is expected.

## Configure local generation

1. Copy `examples/operator.example.json` to a private, ignored working location. Replace every `<NAME>` path placeholder with an absolute path in your local Linux / WSL2 environment. The SAM3 and FLUX Python paths may point to separate environments. Use a dedicated DA3 depth lock path.
2. Install the declared model revisions and binaries using [GPU setup](GPU_SETUP.md). Keep the two DA3 source checkouts at their respective declared revisions. Models, licenses, gated download access, and runtime compatibility are the operator's responsibility.
3. Run the asset helper in that same environment to fill the file hashes and produce the FLUX receipt. Paths and public revisions must already be filled; hash placeholders can remain at this step.

```sh
python scripts/pin_model_assets.py \
  --settings /srv/streetview-to-ply/operator.paths.json \
  --flux-model /srv/models/flux-klein \
  --flux-receipt /srv/streetview-to-ply/flux-receipt.json \
  --output /srv/streetview-to-ply/operator.pinned.json
```

Use fresh output filenames. This helper reads installed files, checks the configured DA3 Git revisions, and hashes the local assets. It performs no download, model inference, or training. The FLUX receipt's revision is operator-declared; local hashing alone does not prove that files came from that upstream revision. A complete `Flux2KleinPipeline` directory with ordinary files is required. The RT-DETR fingerprint includes absolute paths, so regenerate it after moving that model.

4. Save the pinned settings as `.local/operator.json` in this checkout. Keep the FLUX receipt at its configured local path.
5. Activate the GPU Python environment and register the local backend and initial preset:

```sh
python scripts/configure_backend.py \
  --settings .local/operator.json \
  --data-dir tools/streetview_app/data \
  --local
```

This command rejects unresolved placeholders, checks the pipeline configuration, creates `backend.json`, registers a scene-free preset, and sets the default preset in `ui_preferences.json`. It does not start inference or training. Registration binds the current application code and local operator JSON; each stage checks actual assets before using them. It is not a successful-generation receipt.

Restart the GUI after configuration. For later changes, stop active jobs, review the settings, and rerun with `--replace`. Existing job outputs and recipes are retained. Code or operator-file changes invalidate older recipe bindings, so register a new preset after updating the checkout. Do not put `.local`, generated backend settings, SSH keys, model receipts, or job data into Git.

To run the GPU stages with a different local Python environment, add
`--python /absolute/path/to/.venv-gpu/bin/python`. Keep the GUI server running
during generation. Windows users should run both configuration and the GUI
inside WSL2 for this recipe.

<details>
<summary>Optional: use a remote GPU</summary>

Configure the SSH worker in [GPU setup](GPU_SETUP.md), use asset paths on that
worker, and replace `--local` with `--host YOUR_ALIAS --remote-root /srv/streetview-to-ply`.
This is optional; the local backend does not require SSH.

</details>

## Default pipeline

`examples/preset.example.json` contains only portable processing settings. Capture IDs, coordinates, dates, and scene-specific caches are supplied by a real map selection, never by a preset bootstrap.

| Stage | Default behavior |
| --- | --- |
| Collection | Native six cube faces; group cameras by physical capture station. |
| Object masks | SAM3 on native cube faces, checked against independent RT-DETRv2 boxes; semantic model for broad sky evidence. Object inference projection is `native_cubes`. |
| Panorama removal | Project masks into a complete 2:1 panorama; FLUX at 2048 × 1024, 4 steps, guidance 1.0, deterministic station-derived seeds. Shadows are not added to the removal mask. |
| Composition | Shared normalized panorama transform; 4-pixel feather at width 2048; mask-only composition directly on each original cube grid. Pixels outside the alpha support remain exactly original. |
| SfM | Native 90° perspective cube cameras, 8 candidate neighbors, at least 3 physical stations, full requested registration, actual triangulated XYZRGB seeds. Generated regions and sky do not supply SfM matches. GPS is an auxiliary prior with horizontal sigma 3 m and vertical sigma 20 m. |
| Brush | Brush 0.3, 40,000 steps, maximum 2,000,000 splats, training maximum dimension 1280, SH degree 2; refinement every 250 steps and growth stop at 30,000. |
| Refinement | gsplat, fixed population, 6,000 steps, learning-rate multiplier 0.1, opacity penalty 0.03, scale penalty 0.05, actual SfM inverse-depth loss weight 0.002. |
| Depth cleanup | DA3 at 504 pixels plus semantic sky votes; uncertain non-sky depth can abstain from a deletion vote. No learned Gaussian initialization. |
| Size filter | Delete complete Gaussian rows when their largest standard deviation reaches 0.5 times the physical-camera radius. |

The refinement constants are implemented in `tools/streetview_engine/brush_refine.py`; editing the GUI's generic step field does not silently change this fixed two-stage recipe. All selected stations are used for training (`holdout_fraction=0`, quality comparison disabled). This default does not produce a held-out quality score or establish performance on other locations.

Native image dimensions are preserved in the prepared dataset. Training uses a maximum dimension of 1280. Newly generated content comes from the 2048 × 1024 panorama, so native output dimensions do not imply native-detail reconstruction inside removed objects.

## Shared panorama transform

`panorama_preprocess.common_transform` is a 2 × 3 affine map from original normalized ERP pixel centers to generated normalized ERP pixel centers. The shipped default is the identity:

```json
[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
```

The generation path enforces the same full-field 2:1 input/output size and disables automatic conditioning-image resizing. One matrix is applied to every selected panorama, before sampling generated pixels onto the native cube grid. There are no per-image or per-face fits. This is a source-defined identity transform; it needs no private calibration image or scene-specific calibration receipt. It does not claim to correct every deformation an image generator can introduce.

## Distinct cleanup controls

The default includes `processing_options.remove_sky=false`, `mask_dynamic=true`, `depth_cleanup.enabled=true`, and `size_filter.enabled=true`.

- **Remove sky from training** changes the RGB training mask. It is separate from post-training sky cleanup.
- **Remove people/vehicles** enables the native-mask → whole-panorama removal → mask-only composition path. Generated pixels are usable appearance targets but are excluded from SfM geometry evidence.
- **Depth cleanup** toggles DA3 inference and the combined depth/semantic deletion stage. It does not change the 0.002 SfM training-depth loss or the size filter.
- **Size filter** deletes oversized Gaussian rows. The camera radius is the maximum distance from the equally weighted physical-station centroid, with duplicate face centers grouped together. It is not a depth-map threshold.

Depth cleanup requires at least 3 sky-voting stations, sky/non-sky ratio 0.25, erosion radius 2 pixels, depth factor 1.5, and soft size ratio 0.1. It protects Down-supported rows and rows below the nearest 3 physical camera stations. It does not invent a ground plane. The final independent size cap can still delete a protected oversized row. Depth values are heuristic supporting evidence, not universally validated per-pixel metric measurements.

## CLI

The CLI uses the same running local server and selection checks as the GUI:

```sh
python -m tools.streetview_app.workflow_cli presets
python -m tools.streetview_app.workflow_cli generate --config selection.json --recipe-id RECIPE_ID
python -m tools.streetview_app.workflow_cli status JOB_ID
```

Use a real saved selection from the GUI; do not invent panorama metadata or claim that an unverified selection is verified. The server rechecks provider metadata before generation. Radius and capture-count upper limits are not imposed by the GUI, but actual provider coverage, successful camera registration, memory, and runtime still determine feasible jobs.

The retired single-panorama UniSHARP generation mode is not part of this release.
