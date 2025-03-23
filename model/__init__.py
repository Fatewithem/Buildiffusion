from config.structured import ProjectConfig
from .model import ConditionalPointCloudDiffusionModel
from .model_utils import set_requires_grad
from model.transformer.model_query import MultiModalQueryModel
from model.recolor.model_color import ImageSupervisedPointCloudModel

from omegaconf import DictConfig, OmegaConf

from omegaconf import OmegaConf

# from .mask_loss import get_mask_loss


def get_model(cfg: ProjectConfig):
    model = ConditionalPointCloudDiffusionModel(**cfg.model)
    # if cfg.run.freeze_feature_model:
    #     set_requires_grad(model.feature_model, False)
    return model


def get_color_model(cfg: ProjectConfig):
    model = PointCloudColoringModel(**cfg.model)
    # if cfg.run.freeze_feature_model:
    #     set_requires_grad(model.feature_model, False)
    return model


def get_plane_model(cfg: ProjectConfig):
    # 加载配置文件
    cfg_path = "/home/code/Buildiffusion/config/config_plane/legoformer_s.yaml"  # 传入你当前的配置文件路径
    cfg = load_config(cfg_path)

    # 输出合并后的配置
    # print(OmegaConf.to_yaml(cfg))
    model = MultiModalQueryModel(config=cfg)

    return model


def get_color_model(cfg: ProjectConfig):
    model = ImageSupervisedPointCloudModel()

    return model


def load_config(cfg_path: str) -> DictConfig:
    """
        Load configuration file. `base_config.yaml` is taken as a base for every config.
    :param cfg_path: Path to the configuration file
    :return: Loaded configuration
    """
    # 加载基础配置文件 base_config.yaml
    base_cfg = OmegaConf.load('/home/code/Buildiffusion/config/config_plane/base_config.yaml')

    # 加载当前配置文件
    curr_cfg = OmegaConf.load(cfg_path)

    # 合并基础配置与当前配置，当前配置会覆盖基础配置中的相同项
    return OmegaConf.merge(base_cfg, curr_cfg)
