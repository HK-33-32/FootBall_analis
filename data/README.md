# Data

Raw match video is never committed. Runtime uploads, model caches and perception artifacts are ignored by Git.

SoccerNet Game State Reconstruction requires the official SoccerNet download flow and acceptance of its terms. Keep the frozen test list under `data/splits/`, but do not use it for prompt or threshold tuning. Development work should use train/validation or the already-computed legacy runs supplied outside this repository.

Soccer Factory is published by the SoccerMaster authors as separate video and annotation archives. Record checksums, release URL and license terms next to any local download before including it in a benchmark run.

