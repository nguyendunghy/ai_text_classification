from pathlib import Path

gpus = [0]
batch_size = 24
lr = 2e-5

epochs = 100
num_workers = 16

resources = Path('./resources')

model_names = [
    None,
    'casperhansen/llama-3-8b-instruct-awq',
    'TheBloke/Mistral-7B-Instruct-v0.2-AWQ',
    'casperhansen/gemma-7b-it-awq',
    'TheBloke/neural-chat-7B-v3-3-AWQ',
    'TheBloke/zephyr-7B-beta-AWQ',
    'TheBloke/OpenHermes-2.5-Mistral-7B-16k-AWQ',
    'TheBloke/WizardCoder-33B-V1.1-AWQ',
]
model_names = {model_name: idx for idx, model_name in enumerate(model_names)}


def datamodule_cfg():
    return dict(
        loader_kwargs=dict(
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        dataset_cfg=dict(
            type='PKLDataset',
            csv_file='resources/data.pkl',
            model_names=model_names,
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
                 filename='checkpoint_{BinaryAccuracy:.3f}_{MulticlassAccuracy:.3f}')
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
            num_model_names=len(model_names),
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
