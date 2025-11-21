import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch.nn as nn

from hydra.core.config_store import ConfigStore
from hydra.conf import RunDir


@dataclass
class CustomHydraRunDir(RunDir):
    dir: str = './outputs/${run.name}/${now:%Y-%m-%d--%H-%M-%S}'


@dataclass
class RunConfig:
    name: str = 'debug'
    mode: str = 'sample'  # sample train
    mixed_precision: str = 'fp16'  # no
    cpu: bool = False
    seed: int = 42
    val_before_training: bool = False
    vis_before_training: bool = False
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None
    max_steps: int = 300_000
    checkpoint_freq: int = 5000
    val_freq: int = 5_000
    vis_freq: int = 5_000
    log_step_freq: int = 20
    print_step_freq: int = 100

    # Inference config
    num_inference_steps: int = 1000
    diffusion_scheduler: Optional[str] = 'ddpm'
    num_samples: int = 1
    num_sample_batches: Optional[int] = None
    sample_from_ema: bool = False
    sample_save_evolutions: bool = True  # temporarily set by default

    # Training config
    freeze_feature_model: bool = True

    # Coloring training config
    coloring_training_noise_std: float = 0.0
    coloring_sample_dir: Optional[str] = None


@dataclass
class LoggingConfig:
    wandb: bool = False
    wandb_project: str = 'bd'


@dataclass
class PointCloudProjectModelConfig:
    # Feature extraction arguments
    image_height: int = '${dataset.image_height}'
    image_width: int = '${dataset.image_width}'
    image_feature_model: str = 'vit_base_patch16_224_mae'  # or 'vit_base_patch16_224_mae' or 'identity'
    use_local_colors: bool = True
    use_local_features: bool = True
    use_global_features: bool = False
    use_mask: bool = True
    use_top: bool = True
    use_depth: bool = True
    use_distance_transform: bool = True

    # Point cloud data arguments. Note these are here because the processing happens
    # inside the model, rather than inside the dataset.
    scale_factor: float = "${dataset.scale_factor}"
    colors_mean: float = 0.5
    colors_std: float = 0.5
    color_channels: int = 3
    predict_shape: bool = True
    predict_color: bool = False


@dataclass
class PointCloudDiffusionModelConfig(PointCloudProjectModelConfig):
    # Diffusion arguments
    beta_start: float = 1e-5  # 0.00085
    beta_end: float = 8e-3  # 0.012
    beta_schedule: str = 'linear'  # 'custom' 'linear'

    # Point cloud model arguments
    point_cloud_model: str = 'pvcnnplusplus'  # pvcnnplusplus
    point_cloud_model_embed_dim: int = 512


@dataclass
class PointCloudColoringModelConfig(PointCloudProjectModelConfig):
    # Projection arguments
    predict_shape = True
    predict_color = False

    # Point cloud model arguments
    point_cloud_model: str = 'pvcnn'
    point_cloud_model_layers: int = 1
    point_cloud_model_embed_dim: int = 512


@dataclass
class DatasetConfig:
    type: str


@dataclass
class PointCloudDatasetConfig(DatasetConfig):
    eval_split: str = 'test'
    max_points: int = 150_000
    image_height: int = 1080
    image_width: int = 1920
    scale_factor: float = 10.0
    buildings_ids: Optional[List] = field(default_factory=list)


@dataclass
class BuildingsConfig(PointCloudDatasetConfig):
    type: str = 'buildings'
    root: str = '/home/datasets/UrbanBIS'
    scene: List[str] = field(default_factory=lambda: ['Qingdao', 'Lihu', 'Longhua'])
    mask_images: bool = '${model.use_mask}'
    top_image: bool = '${model.use_top}'
    depth_images: bool = '${model.use_depth}'


@dataclass
class AugmentationConfig:
    pass


@dataclass
class DataloaderConfig:
    batch_size: int = 1  # shapenet: 16/4 / hydrant: 2 / buliding: 6 / teddybear: 8 / toytruck : 3
    num_workers: int = 12  # shapenet : 4 / co3d: 12 / building: 12
    # 0 for debug  32


@dataclass
class LossConfig:
    diffusion_weight: float = 1.0
    rgb_weight: float = 1.0
    consistency_weight = 1.0


@dataclass
class CheckpointConfig:
    resume: Optional[str] = None
    resume_training: bool = True
    resume_training_optimizer: bool = True
    resume_training_scheduler: bool = True
    resume_training_state: bool = True


@dataclass
# 控制和调整模型训练中的 EMA 行为
class ExponentialMovingAverageConfig:
    use_ema: bool = False
    # # From Diffusers EMA (should probably switch)
    # ema_inv_gamma: float = 1.0
    # ema_power: float = 0.75
    # ema_max_decay: float = 0.9999
    decay: float = 0.999
    update_every: int = 20


@dataclass
class OptimizerConfig:
    type: str
    name: str
    lr: float = 1e-5
    weight_decay: float = 0.0
    scale_learning_rate_with_batch_size: bool = False
    gradient_accumulation_steps: int = 1
    clip_grad_norm: Optional[float] = 50.0  # 5.0
    kwargs: Dict = field(default_factory=lambda: dict())  # 额外参数的字典，允许在初始化优化器时传入其他特定的配置


@dataclass
class AdadeltaOptimizerConfig(OptimizerConfig):
    type: str = 'torch'
    name: str = 'Adadelta'
    kwargs: Dict = field(default_factory=lambda: dict(
        weight_decay=5e-4,
    ))


@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    type: str = 'torch'
    name: str = 'AdamW'
    weight_decay: float = 1e-5
    kwargs: Dict = field(default_factory=lambda: dict(betas=(0.95, 0.999)))


@dataclass
class SchedulerConfig:
    type: str
    kwargs: Dict = field(default_factory=lambda: dict())


@dataclass
class LinearSchedulerConfig(SchedulerConfig):
    type: str = 'transformers'
    kwargs: Dict = field(default_factory=lambda: dict(
        name='linear',
        num_warmup_steps=0,
        num_training_steps="${run.max_steps}",
    ))


@dataclass
class CosineSchedulerConfig(SchedulerConfig):
    type: str = 'transformers'
    kwargs: Dict = field(default_factory=lambda: dict(
        name='cosine',
        num_warmup_steps=2000,  # 0
        num_training_steps="${run.max_steps}",
    ))


@dataclass
class MVSFormerWithDinoConfig:
    dino_model_path: str = "/home/code/Buildiffusion/cost_volume/dinov2_base"
    transformer_config: Optional[List[Dict[str, Any]]] = None


@dataclass
class ProjectConfig:
    run: RunConfig
    logging: LoggingConfig
    dataset: PointCloudDatasetConfig
    augmentations: AugmentationConfig
    dataloader: DataloaderConfig
    loss: LossConfig
    model: PointCloudProjectModelConfig
    mvsmodel: MVSFormerWithDinoConfig
    ema: ExponentialMovingAverageConfig
    checkpoint: CheckpointConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    defaults: List[Any] = field(default_factory=lambda: [
        'custom_hydra_run_dir',
        {'run': 'default'},
        {'logging': 'default'},
        {'model': 'diffrec'},
        {'mvsmodel': 'default'},
        {'dataset': 'buildings'},
        {'augmentations': 'default'},
        {'dataloader': 'default'},
        {'ema': 'default'},
        {'loss': 'default'},
        {'checkpoint': 'default'},
        {'optimizer': 'adam'},
        {'scheduler': 'cosine'},
    ])


cs = ConfigStore.instance()
cs.store(name='custom_hydra_run_dir', node=CustomHydraRunDir, package="hydra.run")
cs.store(group='run', name='default', node=RunConfig)
cs.store(group='logging', name='default', node=LoggingConfig)
cs.store(group='model', name='diffrec', node=PointCloudDiffusionModelConfig)
cs.store(group='model', name='coloring_model', node=PointCloudColoringModelConfig)
cs.store(group="mvsmodel", name="default", node=MVSFormerWithDinoConfig)
cs.store(group='dataset', name='buildings', node=BuildingsConfig)
cs.store(group='augmentations', name='default', node=AugmentationConfig)
cs.store(group='dataloader', name='default', node=DataloaderConfig)
cs.store(group='loss', name='default', node=LossConfig)
cs.store(group='ema', name='default', node=ExponentialMovingAverageConfig)
cs.store(group='checkpoint', name='default', node=CheckpointConfig)
cs.store(group='optimizer', name='adadelta', node=AdadeltaOptimizerConfig)
cs.store(group='optimizer', name='adam', node=AdamOptimizerConfig)
cs.store(group='scheduler', name='linear', node=LinearSchedulerConfig)
cs.store(group='scheduler', name='cosine', node=CosineSchedulerConfig)
cs.store(name='config', node=ProjectConfig)

























