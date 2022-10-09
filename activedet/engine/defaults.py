import os
import torch
from detectron2.engine.defaults import default_argument_parser, _try_get_key, _highlight
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.utils.env import seed_all_rng
from detectron2.config import CfgNode, LazyConfig
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.file_io import PathManager


def faster_setup(cfg, args):
    """Perform some basic common setups at the beginning of a job.
    It's similar to default_setup. However, we remove saving of config
    to speed up lazy trainer. 
    Additionally, we remove higlights on config.py to provide human-readable
    config.py on log.txt

    Lastly, we only use this on LazyTrainer
    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = _try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    setup_logger(output_dir, distributed_rank=rank, name="detectron2")
    logger = setup_logger(output_dir, distributed_rank=rank, name="activedet")


    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                PathManager.open(args.config_file, "r").read(),
            )
        )
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        if isinstance(cfg, CfgNode):
            logger.info("Running with full config:\n{}".format(cfg.dump()))
            with PathManager.open(path, "w") as f:
                f.write(cfg.dump())
        else:
            LazyConfig.save(cfg, path)
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed = _try_get_key(cfg, "SEED", "train.seed", default=-1)
    seed_all_rng(None if seed < 0 else seed + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = _try_get_key(
            cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False
        )


def parser_with_acquisition(epilog=None):
    """Adds the custom arguments applied for active learning training

    Args:
        epilog (str, optional): epilog passed to ArgumentParser describing the usage.. Defaults to None.
    """
    parser = default_argument_parser(epilog)

    parser.add_argument("--acquire-only", action="store_true", help="perform pool evaluation only")

    return parser