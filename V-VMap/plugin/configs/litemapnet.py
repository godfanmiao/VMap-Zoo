_base_ = [
    './_base_/default_runtime.py'
]

# model type
type = 'Mapper'
plugin = True

# plugin code dir
plugin_dir = 'plugin/'

# img configs
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_h = 480
img_w = 800
img_size = (img_h, img_w)

num_gpus = 2
batch_size = 4
num_iters_per_epoch = 27846 // (num_gpus * batch_size)
num_epochs = 400
total_iters = num_iters_per_epoch * num_epochs

num_queries = 100

# category configs
cat2id = {
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
}
num_class = max(list(cat2id.values()))+1

# bev configs+
roi_size = (60, 30)

# vectorize params
coords_dim = 2
sample_dist = -1
sample_num = -1
simplify = True

# meta info for submission pkl
meta = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
    output_format='vector')

# model configs
bev_embed_dims = 256
num_points = 20
permute = True

model = dict(
    type='LiteMapNet',
    num_classes=num_class,
    num_queries=num_queries,
    sync_cls_avg_factor=True,
    pretrained = '/data/yaoyi/HDMap/Litemapnet/pretrained/checkpoint-799.pth',
    backbone_cfg=dict(
        type='ConvNeXtV2',
        depths=[2, 2, 6, 2],
        dims=[40, 80, 160, 320],
        out_indices=(1,2,3)
    ),
    neck_cfg=dict(
        type='FPN',
        in_channels=[320, 160, 80],
        out_channels=bev_embed_dims
    ),
    ipm_cfg=dict(
        type='IPM',
        xbound=[-30.0, 30.0, 0.3],
        ybound=[-15.0, 15.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        out_channels=bev_embed_dims
    ),
    head_cfg=dict(
        type='MapDetectorHead',
        num_classes=num_class,
        in_channels=bev_embed_dims,
        embed_dims=bev_embed_dims,
        num_queries=num_queries,
        num_points = num_points
    ),
    loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=4.0
        ),
    loss_reg=dict(
        type='LinesL1Loss',
        loss_weight=50.0,
        beta=0.01,
    ),
    assigner=dict(
        type='HungarianLinesAssigner',
        cost=dict(
            type='MapQueriesCost',
            cls_cost=dict(type='FocalLossCost', weight=4.0),
            reg_cost=dict(type='LinesL1Cost', weight=50.0, beta=0.01, permute=permute),
        ),
    ),
)

# data processing pipelines
train_pipeline = [
    dict(
        type='VectorizeMap',
        coords_dim=coords_dim,
        roi_size=roi_size,
        sample_num=num_points,
        normalize=True,
        permute=permute,
    ),
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='ResizeMultiViewImages',
         size=img_size, # H, W
         change_intrinsics=True,
         ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img', 'vectors'], meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'))
]

# data processing pipelines
test_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='ResizeMultiViewImages',
         size=img_size, # H, W
         change_intrinsics=True,
         ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img'], meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'))
]

# configs for evaluation code
# DO NOT CHANGE
eval_config = dict(
    type='NuscDataset',
    data_root='/data/yaoyi/HDMap/Litemapnet/datasets/nuscenes',
    ann_file='/data/yaoyi/HDMap/Litemapnet/datasets/nuscenes/nuscenes_map_infos_val.pkl',
    meta=meta,
    roi_size=roi_size,
    cat2id=cat2id,
    pipeline=[
        dict(
            type='VectorizeMap',
            coords_dim=coords_dim,
            simplify=True,
            normalize=False,
            roi_size=roi_size
        ),
        dict(type='FormatBundleMap'),
        dict(type='Collect3D', keys=['vectors'], meta_keys=['token'])
    ],
    interval=1,
)

# dataset configs
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type='NuscDataset',
        data_root='/data/yaoyi/HDMap/Litemapnet/datasets/nuscenes',
        ann_file='/data/yaoyi/HDMap/Litemapnet/datasets/nuscenes/nuscenes_map_infos_train.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=train_pipeline,
        seq_split_num=1,
    ),
    val=dict(
        type='NuscDataset',
        data_root='/data/yaoyi/HDMap/Litemapnet/datasets/nuscenes',
        ann_file='/data/yaoyi/HDMap/Litemapnet/datasets/nuscenes/nuscenes_map_infos_val.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=1,
    ),
    test=dict(
        type='NuscDataset',
        data_root='/data/yaoyi/HDMap/Litemapnet/datasets/nuscenes',
        ann_file='/data/yaoyi/HDMap/Litemapnet/datasets/nuscenes/nuscenes_map_infos_val.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=1,
    ),
    shuffler_sampler=dict(
        type='InfiniteGroupEachSampleInBatchSampler',
        seq_split_num=2,
        num_iters_to_seq=num_epochs//6*num_iters_per_epoch*num_iters_per_epoch,
        random_drop=0.0
    ),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=5e-4 * (batch_size / 4),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
        }),
    weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy & schedule
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=3e-3)

evaluation = dict(interval=num_epochs//6*num_iters_per_epoch)
find_unused_parameters = True #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=num_epochs//6*num_iters_per_epoch)

runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

SyncBN = True
