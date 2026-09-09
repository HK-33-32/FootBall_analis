# Third-party components and data

The Apache-2.0 license in this repository applies to original application and
Football Core code and documentation owned by the project owner. It does not
grant rights to external model weights, datasets, match footage, container base
images or vendored third-party code under another license.

The full GPU profile builds from `services/perception` and expects model
checkpoints mounted under `/opt/weights`. Checkpoints are deliberately absent;
obtain and use each under its own terms.

The perception tree contains modified components derived from YOLOX (Apache-2.0)
and deep-person-reid (MIT; its license is retained at
`services/perception/engine/reid/LICENSE`). Deep-EIoU tracker attribution is
preserved in source/history; verify any upstream model/data terms separately.

Named research dependencies include SoccerNet, Qwen, CLIP/OpenAI, RF-DETR and
SoccerMaster. Consult their official repositories/model cards and the dependency
metadata installed by Python or Docker for current terms. Raw SoccerNet data and
football broadcasts must not be committed here.
