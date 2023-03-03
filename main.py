import glob
import os
import pickle

import cv2
import hydra
import numpy as np
import torchvision.transforms as T
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from VAE.models.VAE import VAE
from pytorch_lightning import loggers as pl_loggers
from VAE.DataLoaders.LeafDataLoader import LeafDataLoader
import pytorch_lightning as pl

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    transform = T.Compose([
                T.Resize(size=(cfg.input_size, cfg.input_size)),
                T.ToTensor()])
    datamodule = LeafDataLoader(cfg.dataset_path,  cfg.image_path, batch_size=cfg.batch_size,transform=transform )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_epoch",
        dirpath=cfg.checkpoint_path,
        filename="{epoch:02d}-{val_loss:.2f}",
        save_last=True,
        mode="min"
    )

    model = VAE(cfg, cfg.input_channels, cfg.latent_space_size)

    if cfg.restore_from_checkpoint:
        list_of_files = glob.glob(cfg.checkpoint_path+"*")  # * means all if need specific format then *.csv
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
        model = model.load_from_checkpoint(latest_checkpoint)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.logdir, name=str(cfg.input_size)+"/"+str(cfg.kld_weight)+"/"+str(cfg.latent_space_size))
    trainer = pl.Trainer(max_epochs=cfg.epochs,
                         accelerator="gpu",
                         devices=1,
                         logger=tb_logger,
                         callbacks=[EarlyStopping("val_loss_epoch", patience=cfg.early_stopping_patience),
                                    checkpoint_callback])

    if cfg.Train:
        trainer.fit(model, datamodule)

    if cfg.Test:
        predictions = trainer.predict(model, datamodule)


        if not os.path.exists(cfg.prediction_logdir):
            os.makedirs(cfg.prediction_logdir)
        with open(cfg.prediction_logdir+"/prediction.pkl", "wb") as f:
            pickle.dump(predictions, f)


if __name__ == "__main__":
    main()