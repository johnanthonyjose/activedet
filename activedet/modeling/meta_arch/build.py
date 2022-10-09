from detectron2.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip

from detectron2.modeling import build_model as d2_build_model

from .model_aux import build_meta_arch_aux




def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    try:
        # Check if AUX exists in config
        if cfg.MODEL.AUX.NAME is not None:
            return build_meta_arch_aux(cfg)
    except AttributeError:
        pass
    return d2_build_model(cfg)
