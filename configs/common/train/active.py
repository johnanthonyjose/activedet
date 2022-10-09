train = dict(
    output_dir="./output",
    init_checkpoint=None,
    max_iter=87000,
    model_ema=dict(enabled=False,
                    decay_rate=0.9998), #EfficientDet default momentum
    amp=dict(enabled=True),  # options for Automatic Mixed Precision
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    checkpointer=dict(max_to_keep=100),  # options for PeriodicCheckpointer
    log_period=20,
    init_checkpoint_path="model_initial",
    detach_points = None,
    device="cuda"
    # ...
)