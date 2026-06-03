import os
from datetime import datetime

import torch
import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig
from omegaconf import OmegaConf

from model_training.dataset import create_dataloaders
from model_training.lightning_module import SegmentationLightningModule
from model_training.predictor import SODPredictor
from model_training.compute_metrics import process_dataset

OmegaConf.register_new_resolver("eval", eval)


class EvaluationCallback(pl.Callback):
    def __init__(self, eval_config):
        super().__init__()
        self.eval_config = eval_config
        self.datasets = eval_config.datasets

    def on_fit_end(self, trainer, pl_module):
        if not self.eval_config.enabled:
            return

        best_model_path = trainer.checkpoint_callback.best_model_path
        predictor = SODPredictor(
            best_model_path,
            self.eval_config.image_size,
            device='cuda'
        )

        all_metrics = {}
        for dataset in self.datasets:
            data_input_dir = os.path.join(self.eval_config.input_dir, dataset)
            metrics = process_dataset(data_input_dir, predictor)
            all_metrics[dataset] = metrics
            print(f"\n{dataset} metrics:")
            print(metrics)

            if isinstance(trainer.logger, pl.loggers.TensorBoardLogger):
                # For metrics dictionary
                for dataset_name, metrics in all_metrics.items():
                    for metric_name, value in metrics.items():
                        trainer.logger.experiment.add_scalar(
                            f"evaluation/{dataset_name}/{metric_name}",
                            value,
                            global_step=trainer.global_step
                        )


def get_experiment_name(base_name: str) -> str:
    """
    Create a unique experiment name by combining the base name with a timestamp.

    Args:
        base_name: Base experiment name from config

    Returns:
        str: Unique experiment name with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"


@hydra.main(config_path="config", config_name="train")
def train(config: DictConfig):
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(config.backend.seed)
    experiment_name = get_experiment_name(config.train_stage.experiment_name)

    train_loader, val_loader = create_dataloaders(
        dataset_paths=config.dataset.datasets,
        image_size=config.dataset.image_size,
        train_batch_size=config.dataset.train_batch_size,
        val_batch_size=config.dataset.val_batch_size,
        num_workers=config.dataset.num_workers,
        val_split=config.dataset.val_split,
        seed=config.backend.seed,
        transform_mode=config.dataset.transform_mode,
        flux_features_dir=config.dataset.get('flux_features_dir', None),
        feature_layers=config.dataset.get('feature_layers', [0, 1, 2, 3]),
        debug_subset_fraction=config.dataset.get('debug_subset_fraction', None),
    )
    
    model = SegmentationLightningModule(config)

    logger = TensorBoardLogger(
        save_dir=config.train_stage.log_dir,
        name=experiment_name,
        default_hp_metric=False
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config.train_stage.save_dir}/{experiment_name}",
        filename='{epoch:02d}-{val_dice_epoch:.4f}',
        monitor='val_dice_epoch',
        mode='max',
        save_top_k=3,
        save_last=True
    )

    callbacks = [
        checkpoint_callback,
        EarlyStopping(**config.train_stage.early_stopping),
        EvaluationCallback(config.train_stage.evaluation),
        LearningRateMonitor(logging_interval='step')
    ]

    trainer = pl.Trainer(
        accelerator=config.backend.accelerator,
        devices=config.backend.devices,
        max_epochs=config.backend.max_epochs,
        precision=config.backend.precision,
        accumulate_grad_batches=config.backend.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        strategy=config.backend.get('strategy', 'fsdp'),
    )
    if config.train_stage.checkpoint_path is not None:
        if config.train_stage.weights_only:

            state_dict = SegmentationLightningModule.load_from_checkpoint(
                config.train_stage.checkpoint_path,
                weights_only=True,
            ).state_dict()
            model.load_state_dict(state_dict)
            trainer.fit(model, train_loader, val_loader)
        else:
            trainer.fit(model, train_loader, val_loader, ckpt_path=config.train_stage.checkpoint_path)
    else:
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
