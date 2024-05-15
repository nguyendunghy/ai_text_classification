from pathlib import Path

# seed = 42
gpus = [0]
batch_size = 24
lr = 2e-4

epochs = 10
num_workers = 16

resources = Path('./resources')


def datamodule_cfg():
    return dict(
        loader_kwargs=dict(
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        dataset_cfg=dict(
            type='PKLDataset',
            csv_file='resources/data.pkl'
        )
    )


def trainer_cfg(**kwargs):
    return dict(
        max_epochs=epochs,
        min_epochs=epochs,
        callbacks=[
            dict(type='LearningRateMonitor', logging_interval='step'),
            dict(type='ModelCheckpoint', save_top_k=3, save_last=True, verbose=True, mode='max',
                 monitor='BinaryAccuracy', dirpath=resources / 'checkpoints',
                 filename='checkpoint_{BinaryAccuracy:.3f}')
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
                 f'_bs{batch_size}_epochs{epochs}',
            project='text_ai_classification',
            key_path=Path('configs/wandb.config')
            # mode='disabled'
        )
    )


def mainmodule_cfg():
    return dict(
        type='BaselineClassificator',
        debug=False,
        # Model agnostic parameters
        backbone_cfg=dict(
            type='DistilBert',
            model_name='distilbert-base-uncased',
            dropout=0.2,
            attention_dropout=0.2,
        ),
        tokenizer_cfg=dict(
            type='DistilBertTokenizer',
            model_name='distilbert-base-uncased',
        ),
        head_cfg=dict(
            type='ClassificationHead',
            in_features=768,
            dropout=0.2,
        ),
        # Optimization stuff agnostic parameters
        optimizer_cfg=dict(
            type='AdamW',
            lr=lr,
            betas=[0.9, 0.999],
            weight_decay=0.05,
            eps=1e-6
        ),
        # scheduler_cfg=dict(
        #     type='CosineScheduleWithWarmup',
        #     num_warmup_steps=100,
        #     num_cycles=1,
        #     num_training_steps=(ds_size // batch_size) * epochs,
        # ),
        # scheduler_update_params=dict(
        #     interval='step',
        #     frequency=1
        # ),
        train_transforms=[],
        metrics=[
            dict(type='Accuracy', task='binary')
        ],
        # eval_score_thresh=0.2,
        # eval_nms_thresh=0.8
    )
