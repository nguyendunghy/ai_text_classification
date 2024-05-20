from pathlib import Path

seed = 0
gpus = [0]
batch_size = 16
lr = 2e-5

epochs = 5
num_workers = 8

resources = Path('./resources')

model_names = [
    None,
    'mistral:text',
    'llama3:text',
    'mixtral:text',
    'gemma:7b',
    'command-r',
    'neural-chat',
    'zephyr:7b-beta',
    'openhermes',
    'wizardcoder',
    'starling-lm:7b-beta',
    'yi:34b',
    'openchat:7b',
    'dolphin-mistral',
    'solar',
    'llama2:13b'
]
model_names = {model_name: idx for idx, model_name in enumerate(model_names)}

backbone_model_name = 'microsoft/deberta-v3-base'
test_datasets = [dict(type='JsonDataset',
                      json_file=str(json_file)) for json_file in Path('resources/sample_data').glob('*.json')]


def datamodule_cfg():
    return dict(
        loader_kwargs=dict(
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        tokenizer_cfg=dict(
            type='AutoTokenizer',
            model_name=backbone_model_name,
            max_seq_len=512,
        ),
        train_dataset_cfg=dict(
            type='PKLDataset',
            csv_file='resources/data.pkl',
            model_names=model_names,
            transforms=[
                dict(type='DataAugmentator'),
            ]
        ),
        val_dataset_cfg=dict(
            type='ConcatDataset',
            datasets=test_datasets[:len(test_datasets) // 2],
        ),
        test_dataset_cfg=dict(
            type='ConcatDataset',
            datasets=test_datasets[len(test_datasets) // 2:],
        )
    )


def trainer_cfg(**kwargs):
    return dict(
        max_epochs=epochs,
        min_epochs=epochs,
        callbacks=[
            dict(type='LearningRateMonitor', logging_interval='step'),
            dict(type='ModelCheckpoint', save_top_k=3, save_last=False, verbose=True, mode='max',
                 monitor='BinaryAccuracy', dirpath=resources / 'checkpoints',
                 enable_version_counter=True,
                 filename='checkpoint'
                          f'_ds{kwargs["ds_size"]}'
                          '_epoch_{epoch:02d}_{BinaryAccuracy:.3f}'),
            # dict(type='DeviceStatsMonitor')
        ],
        benchmark=True,
        accumulate_grad_batches=1,
        log_every_n_steps=10,
        val_check_interval=1.0,
        gradient_clip_val=1.0,
        deterministic=False,
        precision="16-mixed",
        sync_batchnorm=False,
        accelerator='gpu',
        devices=gpus,
        # strategy=DDPStrategy(find_unused_parameters=True),
        wandb_logger=dict(
            name=f'{Path(__file__).stem}'
                 f'_ds{kwargs["ds_size"]}'
                 f'_bs{batch_size}_epochs{epochs}',
            project='text_ai_classification',
            key_path=Path('configs/wandb.config'),
            # mode='disabled'
        ),
        # profiler="simple"
    )


def mainmodule_cfg(**kwargs):
    return dict(
        type='BaselineClassificator',
        debug=False,
        # Model agnostic parameters
        backbone_cfg=dict(
            type='AutoModel',
            model_name=backbone_model_name,
        ),
        head_cfg=dict(
            type='ClassificationHead',
            in_features=768,
            dropout=0.1,
            num_model_names=len(model_names),
            pos_weight=kwargs['pos_weight']
        ),
        # Optimization stuff agnostic parameters
        optimizer_cfg=dict(
            type='AdamW',
            lr=lr,
            betas=[0.9, 0.999],
            weight_decay=0.05,
            eps=1e-8
        ),
        scheduler_cfg=dict(
            num_warmup_steps=0,
            num_training_steps=(kwargs['train_ds_size'] // batch_size) * epochs,
        ),
        metrics=[
            dict(type='Accuracy', task='binary'),
            dict(type='F1Score', task='binary')
        ],
    )
