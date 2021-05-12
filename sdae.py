from ludwig.api import LudwigModel

config = {

}

model = LudwigModel(
  config,
  logging_level=40,
  use_horovod=True,
  gpus=[0,1],
  gpu_memory_limit=None,
  allow_parallel_threads=True
)