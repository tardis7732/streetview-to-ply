# References and acknowledgments

Streetview to PLY combines the projects below with its own map selection,
panorama preparation, job orchestration, reconstruction checks, cleanup, and
export code. Thank you to their authors and maintainers.

## Current reconstruction pipeline

These are the upstream projects and model interfaces used by the connected
`panorama_brush_refine` workflow. Their roles describe this repository's
integration; upstream results and benchmarks do not establish the quality of
this combined workflow.

| Project | Role in Streetview to PLY | Primary source |
|---|---|---|
| streetlevel | Naver panorama discovery, capture metadata, and image retrieval. | [Repository](https://github.com/sk-zk/streetlevel) · [Documentation](https://streetlevel.readthedocs.io/en/master/index.html) |
| Meta Segment Anything Model 3 (SAM 3) | Object masks on native cube faces, with a separate detector checking object evidence. | [Repository](https://github.com/facebookresearch/sam3) · [Transformers adapter](https://huggingface.co/docs/transformers/main/en/model_doc/sam3) |
| RT-DETRv2 | Independent object boxes used to verify SAM 3 object masks. | [Repository](https://github.com/lyuwenyu/RT-DETR) · [Transformers adapter](https://huggingface.co/docs/transformers/main/en/model_doc/rt_detr_v2) |
| Hugging Face Transformers semantic segmentation | Broad sky evidence from an operator-configured semantic model and its actual class labels. | [Semantic segmentation documentation](https://huggingface.co/docs/transformers/tasks/semantic_segmentation) |
| Black Forest Labs FLUX.2 [klein] | Full equirectangular panorama editing through Diffusers `Flux2KleinPipeline`. This project then composites edited pixels only inside the removal masks on the original cube-face grid. | [Repository](https://github.com/black-forest-labs/flux2) · [Diffusers pipeline](https://huggingface.co/docs/diffusers/main/api/pipelines/flux2) |
| COLMAP / PyCOLMAP | Multi-station camera registration and sparse Structure-from-Motion (SfM) points used to initialize training. | [Repository](https://github.com/colmap/colmap) · [Documentation](https://colmap.github.io/) |
| Brush | Initial Gaussian training from SfM seeds. The connected recipe uses Brush 0.3.0 for 40,000 steps. | [Repository](https://github.com/ArthurBrussee/brush) |
| gsplat | Gaussian rasterization and further optimization. The connected recipe uses gsplat 1.5.3 for 6,000 steps with sparse SfM depth supervision at weight 0.002. | [Repository](https://github.com/nerfstudio-project/gsplat) |
| Depth Anything 3 (DA3) | Raw depth from configured metric/nested and known-pose models assists sky-artifact cleanup. Depth serves as evidence for retaining uncertain regions, not as measured ground truth. | [Repository](https://github.com/ByteDance-Seed/Depth-Anything-3) |

The training initialization is always SfM point data. The final cleanup and
size filter are implemented in this repository; no location-specific panorama
IDs or hand-selected Gaussian rows are product defaults. See
[Configuration](CONFIGURATION.md) for the recipe and
[GPU setup](GPU_SETUP.md) for separately installed runtimes and assets.

## Map interface and imagery

The map interface uses [Leaflet](https://leafletjs.com/) and map tiles credited
to [OpenStreetMap contributors](https://www.openstreetmap.org/copyright).
Streetview imagery and the website segment in the demo come from
[NAVER Maps](https://map.naver.com/). These providers are distinct from the
reconstruction models and are not the authors of this tool.

## Optional Unreal preview

[MLSLabsGaussianSplattingRenderer-UE](https://github.com/mlslabs/MLSLabsGaussianSplattingRenderer-UE)
provides the separately installed `MLSLabsRenderer` plugin used by the optional
**Open in Unreal** action. This repository supplies a Python preview adapter.
PLY generation and export do not require Unreal Engine or this plugin.

## Inspiration and earlier research

[YellowO2/streetview-to-3dgs](https://github.com/YellowO2/streetview-to-3dgs)
was the original project inspiration for turning street-view panoramas into
Gaussian scenes. Its published workflow references SHARP, DA3, and FLUX. The
current Streetview to PLY generation path uses multi-station SfM initialization
followed by Brush and gsplat training; it does not reproduce that project's
training pipeline or claim its results.

[SHARP](https://github.com/apple-aiml-research/ml-sharp) and
[UniSHARP](https://github.com/Insta360-Research-Team/UniSHARP) informed earlier
Gaussian-prediction experiments. Predicted Gaussians from these models are
retired as initialization or fusion inputs, and single-panorama UniSHARP
generation is absent from the product GUI, API, and CLI. Existing experimental
outputs are historical records.

The generic backend retains an optional depth-only UniSHARP adapter using the
first-surface geometry of
[UniK3D](https://github.com/lpiccinelli-eth/UniK3D). It is inactive in the named
panorama/Brush recipe and does not expose predicted Gaussian generation.

## External assets

Model weights, raw panorama datasets, external program binaries, Unreal Engine,
and MLSLabsRenderer are not redistributed in this repository. Obtain each
dependency from its own distribution and consult its accompanying code,
model, or asset terms. The links above are acknowledgments and technical
references, not a grant of rights to those external assets.
