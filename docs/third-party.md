# Third-party provenance

- SAM2 source: https://github.com/facebookresearch/sam2, revision `2b90b9f5ceec907a1c18123530e92e794ad901a4`, Apache2.0. The fetched source subset includes the upstream license. The model checkpoint URL and observed download SHA256 are recorded when explicitly acquired. No creator-provided checkpoint SHA256 is fabricated.
- MOSE public metric implementation: https://github.com/henghuiding/MOSE-api, revision `a74b23aa3a31597f487eec10f452e0b9f61347e1`, repository MIT license preserved with the fetched scorer. `metricsv2.py` is loaded unchanged; event reduction is reproduced in the wrapper with direct code-agreement fixtures and stricter prediction validation.
- MOSEv2 dataset: https://huggingface.co/datasets/FudanCVL/MOSEv2/tree/24bd8fc5f5ab824a482b5f6dcf472e3a3ef5c704, metadata and official website specify CC BY-NC-SA4.0 for noncommercial research. A conflicting shorter hub license tag is not treated as commercial permission.
- Python dependencies: exact versions and distribution hashes in the platform-specific requirements files. The project setup does not install a custom project package or use another project's environment.

`artifacts/upstream.json` records source archive and individual source-file hashes. Restore with `scripts/fetch_upstream.py`. That operation retrieves code, never trained model weights or dataset videos.

Research references: [SAM2](https://arxiv.org/abs/2408.00714), [MOSEv2](https://arxiv.org/abs/2508.05630), and [MOSEv1](https://arxiv.org/abs/2302.01872). Our small partial adaptation, local split and fixed corrective-click study are project-specific protocols.
