# AIL_team24

## Infrastructure Setup

1. Launch Google Cloud Platform VM Instance with GPU(Ubuntu 16.04 LTS)
    (https://docs.google.com/document/d/1nFPqxqaeIUjkf5ZvwELhNNlZ3IEiOTHWGuUM-JbYQlY/edit#heading=h.4i8illrwq56d)
2. Setup cuda, docker, nvidia-docker

```
$ path/to/repo/nakahara/sh/install_cuda.sh
$ sudo shutdown -r now
$ path/to/repo/nakahara/sh/install_docker.sh
```

3. Launch Docker Image

```
$ sudo docker build -t ail/gpu-image /path/to_repo
$ sudo nvidia docker run --rm -it ail/gpu-image /bin/bash
```