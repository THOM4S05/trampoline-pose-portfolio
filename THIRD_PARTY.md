# Third-party components

The MIT license in this repository covers this project's original code and
documentation. External libraries, model weights, configuration files and
datasets retain their own licenses and terms.

The inference pipeline integrates pretrained RTMDet-tiny and ViTPose-S through
OpenMMLab. It does not claim to have trained those models.

- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- [MMPose](https://github.com/open-mmlab/mmpose)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMCV](https://github.com/open-mmlab/mmcv)
- [PyTorch](https://github.com/pytorch/pytorch)

Model URLs and checksums are recorded in `models/manifest.json`. Setup downloads
assets from upstream; checkpoint binaries are excluded from Git. The GPU smoke
test downloads an upstream example image whose dataset terms apply. That image
and generated smoke-test media are excluded from the public repository.

The README demonstration uses the project owner's selected RedTuckBacks clip.
The demonstration media is excluded from the software's MIT license; no
permission to reuse identifiable athlete footage is granted by that license.
The separate synthetic example uses generated stick-figure coordinates.
