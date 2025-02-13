# DressRecon: Freeform 4D Human Reconstruction from Monocular Video

### [Website](https://jefftan969.github.io/dressrecon) | [Paper](https://jefftan969.github.io/dressrecon/paper.pdf)

3DV 2025 (Oral)

[Jeff Tan](https://jefftan969.github.io/), 
[Donglai Xiang](https://xiangdonglai.github.io/), 
[Shubham Tulsiani](https://shubhtuls.github.io/),
[Deva Ramanan](https://www.cs.cmu.edu/~deva/),
[Gengshan Yang](https://gengshan-y.github.io/)<br>

![Teaser video](assets/teaser.gif)

## About
DressRecon is a method for freeform 4D human reconstruction, with support for dynamic clothing and human-object interactions. Given a monocular video as input, it reconstructs a time-consistent body model, including shape, appearance, articulation of body+clothing, and 3D tracks. The software is licensed under the MIT license. 

## Release Plan
- [x] Training code
- [ ] Data preprocessing scripts
- [ ] Pretrained checkpoints

## Installation

1. Clone DressRecon
```
git clone https://github.com/jefftan969/dressrecon
cd dressrecon
```

2. Create the environment
```
conda create -y -n dressrecon -c conda-forge python=3.9
conda activate dressrecon
pip install torch==2.4.1
conda install -y -c conda-forge absl-py numpy==1.24.4 tqdm trimesh tensorboard opencv scipy scikit-image matplotlib urdfpy networkx=3 einops imageio-ffmpeg pyrender open3d
pip install pysdf geomloss
pip install -e .
# (Optional) Visualization dependencies
pip install viser
```

3. Install third-party libraries
```
# CUDA kernels for fast dual-quaternion skinning
pip install -e lab4d/third_party/dqtorch
# CUDA kernels for 3D Gaussian refinement
pip install -e lab4d/diffgs/third_party/simple-knn
pip install git+https://github.com/gengshan-y/gsplat-dev.git
```

## Data

We provide two ways to obtain data: download our preprocessed data ([dna-0121_02.zip](https://www.dropbox.com/scl/fi/0z4kgnjtb7ka7ipzlwz9u/dna-0121_02.zip?rlkey=x5j3z2hb7f89nr2fa1mskxz88&st=3ztylftg&dl=0)), or process your own data following the instructions (coming soon!).

<details><summary>Expand to download preprocessed data for sequences in the paper:</summary><p>

Each sequence is about 1.7 GB compressed and 2.3GB uncompressed.

  - [dna-0008_01.zip](https://www.dropbox.com/scl/fi/26u5tpytneortyma54ahj/dna-0008_01.zip?rlkey=jsqnfdcspg9hbhpchnejb2nip&st=d0os9185&dl=0)
  - [dna-0047_01.zip](https://www.dropbox.com/scl/fi/dex3kjruajsw2opvpzjrn/dna-0047_01.zip?rlkey=6wj2wvv2lyhqfzf2f3kkie6lw&st=srv0ogf4&dl=0)
  - [dna-0047_12.zip](https://www.dropbox.com/scl/fi/ptidz11xzzoq34sjj47ps/dna-0047_12.zip?rlkey=g9jpyrw4y2w9mzzewfcqzb7oj&st=71x62tcc&dl=0)
  - [dna-0102_02.zip](https://www.dropbox.com/scl/fi/s30yuw0wk89axalzssrr8/dna-0102_02.zip?rlkey=nu8ayw7i254pmkdj4ba77ls95&st=0yap116n&dl=0)
  - [dna-0113_06.zip](https://www.dropbox.com/scl/fi/ato5i81iwhhfmeuvlg3om/dna-0113_06.zip?rlkey=8c6h95u5475bffdsvpct1ja0j&st=w9isxpp5&dl=0)
  - [dna-0121_02.zip](https://www.dropbox.com/scl/fi/0z4kgnjtb7ka7ipzlwz9u/dna-0121_02.zip?rlkey=x5j3z2hb7f89nr2fa1mskxz88&st=3ztylftg&dl=0)
  - [dna-0123_02.zip](https://www.dropbox.com/scl/fi/66gn6busvbormdw0xipw4/dna-0123_02.zip?rlkey=l0y67vb7m9fvgb0ipu05f9a5s&st=2hhclv20&dl=0)
  - [dna-0128_04.zip](https://www.dropbox.com/scl/fi/ncgo6z0vkreedfytordfn/dna-0128_04.zip?rlkey=kl0k3k49ie0noahlewwvun3vx&st=bnlnm8mv&dl=0)
  - [dna-0133_07.zip](https://www.dropbox.com/scl/fi/i2p4z37zy2jowobmt33ll/dna-0133_07.zip?rlkey=qh04g4c6it64godm1onpiacwu&st=vyn4jhjb&dl=0)
  - [dna-0152_01.zip](https://www.dropbox.com/scl/fi/1u3bcw7zrcbe4vtef2ix8/dna-0152_01.zip?rlkey=tqpq6al5w8qzuo3trk1l911sh&st=xko4x2i3&dl=0)
  - [dna-0166_04.zip](https://www.dropbox.com/scl/fi/ptrmoai1oer25xp4gpchx/dna-0166_04.zip?rlkey=j67349da0ahwud3he83jv1xa3&st=c9ososo9&dl=0)
  - [dna-0188_02.zip](https://www.dropbox.com/scl/fi/un8jihizntmto71bk95p4/dna-0188_02.zip?rlkey=ljo9qc6eminpml5zrttcvlgt8&st=hvnoj19v&dl=0)
  - [dna-0206_04.zip](https://www.dropbox.com/scl/fi/747ditofwgawkzgd015th/dna-0206_04.zip?rlkey=aaqxazey5d3s2nmko5cxi6vck&st=dvrdegy9&dl=0)
  - [dna-0239_01.zip](https://www.dropbox.com/scl/fi/pw3mj4coy04zz76b7scku/dna-0239_01.zip?rlkey=9msklwxe57qf09bwqb44stdrl&st=3f5klrn6&dl=0)

</p></details>

To unzip preprocessed data:
```
mkdir database/processed
cd database/processed
unzip {path_to_downloaded_zip}
cd ../..
```

## Demo

This example shows how to reconstruct a human from a monocular video. To begin, download preprocessed data above or process your own videos.

### Training neural fields

To optimize a body model given an input monocular video:
```
python lab4d/train.py --num_rounds 240 --imgs_per_gpu 96 --seqname {data_sequence_name} --logname {name_of_this_experiment}
```
On a 4090 GPU, 240 optimization rounds should take ~8-9 hours. Checkpoints are saved to `logdir/{seqname}-{logname}`. For faster experiments, you can pass `--num_rounds 40` to train a lower-quality model that's not fully converged yet.

<details><summary>The training command above assumes 24GB of GPU memory. Expand if you have 10GB GPU memory:</summary><p>

```
python lab4d/train.py --num_rounds 240 --imgs_per_gpu 32 --grad_accum 3 --seqname {data_sequence_name} --logname {name_of_this_experiment}
```

</details>

<details><summary>Expand for a description of checkpoint contents:</summary><p>

```
logdir/{seqname}-{logname}
  - ckpt_*.pth         => (Saved model checkpoints)
  - metadata.pth       => (Saved dataset metadata)
  - opts.log           => (Command-line options)
  - params.txt         => (Learning rates for each optimizable parameter)
  - uncertainty/*.npy  => (Per-pixel uncertainty cache for weighted pixel sampling during training)
  - *-fg-gauss.ply     => (Body Gaussians over all optimization iterations)
  - *-fg-proxy.ply     => (Body shape and cameras over all optimization iterations)
  - *-fg-sdf.ply       => (Deformation fields range of influence over all optimization iterations)
```

</p></details>


### Exporting meshes
To extract time-consistent meshes, and render the shape and body+clothing Gaussians:
```
python lab4d/export.py --flagfile=logdir/{seqname}-{logname}/opts.log
```
Results are saved to `logdir/{seqname}-{logname}/export_0000`.

The output directory structure is as follows:
```
logdir/{seqname}-{logname}
  - export_0000
      - render-shape-*.mp4     => (Rendered time-consistent body shapes)
      - render-boneonly-*.mp4  => (Rendered body+clothing Gaussians)
      - render-bone-*.mp4      => (Body+clothing Gaussians, overlaid on top of body shape)
      - fg-mesh.ply            => (Canonical shape exported as a mesh)
      - camera.json            => (Saved camera intrinsics)
      - fg
          - bone/*.ply         => (Time-varying body+clothing Gaussians, exported as meshes)
          - mesh/*.ply         => (Time-varying body shape, exported as time-consistent meshes)
          - motion.json        => (Saved camera poses and time-varying articulations)
  - renderings_proxy
      - fg.mp4                 => (Birds-eye-view of cameras and body shape over all optimization iterations)
```

<details><summary>Expand for scripts to visualize the canonical shape, deformation by body Gaussians only, or deformation by clothing Gaussians only:</summary><p>

```
python lab4d/export.py --flag canonical --flagfile=logdir/{seqname}-{logname}/opts.log
python lab4d/export.py --flag body_only --flagfile=logdir/{seqname}-{logname}/opts.log
python lab4d/export.py --flag cloth_only --flagfile=logdir/{seqname}-{logname}/opts.log
```

</p></details>

### Rendering neural fields
To render RGB, normals, masks, and the other modalities described below:
```
python lab4d/render.py --flagfile=logdir/{seqname}-{logname}/opts.log
```
On a 4090 GPU, rendering each frame at 512x512 resolution should take ~20 seconds. Results are saved to `logdir/{seqname}-{logname}/renderings_0000`. For faster rendering, you can render every N-th frame by passing `--stride <N>` above.

The output directory structure is as follows:
```
logdir/{seqname}-{logname}
  - renderings_0000
      - ref
          - depth.mp4    => (Rendered depth, colorized as RGB)
          - feature.mp4  => (Rendered features)
          - mask.mp4     => (Rendered mask)
          - normal.mp4   => (Rendered normal)
          - rgb.mp4      => (Rendered RGB)
```

<details><summary>Expand to describe additional videos that are rendered for debugging purposes:</summary><p>

```
logdir/{seqname}-{logname}
  - renderings_0000
      - ref
          - eikonal.mp4     => (Rendered magnitude of eikonal loss)
          - gauss_mask.mp4  => (Rendered silhouette of deformation field)
          - ref_*.mp4       => (Rendered input signals, after cropping to tight bounding box and reshaping)
          - sdf.mp4         => (Rendered magnitude of signed distance field)
          - vis.mp4         => (Rendered visibility field)
          - xyz.mp4         => (Rendered world-frame canonical XYZ coordinates)
          - xyz_cam.mp4     => (Rendered camera-frame XYZ coordinates)
          - xyz_t.mp4       => (Rendered world-frame time-t XYZ coordinates)
```

</p></details>

## 3D Gaussian refinement

### Training refined 3D Gaussian model

This step requires a pretrained model from the previous section, which we assume is located at `logdir/{seqname}-{logname}`. To run refinement with 3D Gaussians:
```
bash scripts/train_diffgs_refine.sh {seqname} {logname}
```
On a 4090 GPU, 240 optimization rounds should take ~8-9 hours. Checkpoints are saved to `logdir/{seqname}-diffgs-{logname}`. For faster experiments, you can use `--num_rounds 40` to train a lower-quality model that's not fully converged yet.

<details><summary>The training script above assumes 24GB of GPU memory. Expand if you have 10GB GPU memory:</summary><p>

```
bash scripts/train_diffgs_refine.sh {seqname} {logname} --imgs_per_gpu 4 --grad_accum 4
```

</details>

<details><summary>Expand for a description of checkpoint contents:</summary><p>

```
logdir/{seqname}-{logname}
  - ckpt_*.pth       => (Saved model checkpoints)
  - opts.log         => (Command-line options)
  - params.txt       => (Learning rates for each optimizable parameter)
  - *-all-gauss.ply  => (Body Gaussians over all optimization iterations)
  - *-all-proxy.ply  => (3D Gaussians and cameras over all optimization iterations)
```

</p></details>

### Exporting 3D Gaussians
To produce mesh renderings of the dynamic 3D Gaussians:
```
python lab4d/diffgs/export.py --flagfile=logdir/{seqname}-{logname}/opts.log
```
Results are saved to `logdir/{seqname}-{logname}/export_0000`.

The output directory structure is as follows:
```
logdir/{seqname}-{logname}
  - export_0000
      - fg
          - mesh/*.ply      => (Dynamic 3D Gaussians, exported as meshes)
          - motion.json     => (Saved camera poses and time-varying articulations)
      - camera.json         => (Saved camera intrinsics)
      - fg-mesh.ply         => (Canonical 3D Gaussians)
      - render-shape-*.mp4  => (Mesh-rendered dynamic 3D Gaussians)
  - renderings_proxy
      - all.mp4             => (Birds-eye-view of cameras and body shape over all optimization iterations)
```

### Rendering 3D Gaussians
To render RGB, normals, masks, and the other modalities described below:
```
python lab4d/diffgs/render.py --flagfile=logdir/{seqname}-{logname}/opts.log
```
Results are saved to `logdir/{seqname}-{logname}/renderings_0000`. For faster rendering, you can render every N-th frame by passing `--stride <N>` above.

The output directory structure is as follows:
```
logdir/{seqname}-{logname}
  - renderings_0000
      - ref
          - depth.mp4    => (Rendered depth, colorized as RGB)
          - feature.mp4  => (Rendered features)
          - alpha.mp4    => (Rendered mask)
          - rgb.mp4      => (Rendered RGB)
```

<details><summary>Expand to describe additional videos that are rendered for debugging purposes:</summary><p>

```
logdir/{seqname}-{logname}
  - renderings_0000
      - ref
          - ref_*.mp4  => (Rendered input signals, after cropping to tight bounding box and reshaping)
          - xyz.mp4    => (Rendered world-frame canonical XYZ coordinates)
```

</p></details>

## Acknowledgement
- Our codebase is built upon [Lab4D](https://github.com/lab4d-org/lab4d), thanks for building a comprehensive 4D reconstruction framework!
- Our pre-processing pipeline is built upon the following open-sourced repos: 
  - Human-specific priors: [HMR2.0](https://github.com/shubham-goel/4D-Humans), [Sapiens](https://github.com/facebookresearch/sapiens)
  - Features & correspondence: [DINOv2](https://github.com/facebookresearch/dinov2), [VCNPlus](https://github.com/gengshan-y/VCN)
  - Segmentation: [Track-Anything](https://github.com/gaomingqi/Track-Anything), [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO)

## Bibtex
```
@inproceedings{tan2025dressrecon,
  title={DressRecon: Freeform 4D Human Reconstruction from Monocular Video},
  author={Tan, Jeff and Xiang, Donglai and Tulsiani, Shubham and Ramanan, Deva and Yang, Gengshan},
  booktitle={3DV},
  year={2025}
}
```
