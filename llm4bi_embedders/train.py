import logging
import os
from pathlib import Path, PosixPath

import lightning as L
import torch
import torch.utils.data as data
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner.tuning import Tuner
from transformers import AutoTokenizer

from .src.dataset import DataModule
from .src.lightningMods import LightningMod_DocEmbedder
from .src.utils.general import getargs, upload_checkpoint_to_gcs

torch.set_float32_matmul_precision("medium")

args = getargs()

# Create Logger For file stream and console stream
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

sh = logging.StreamHandler()
fh = logging.FileHandler("main.log")

# Create Formatting
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sh.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(sh)
logger.addHandler(sh)

args_dict = vars(args)

# Seed everything
L.seed_everything(args.seed)

# Setup google Credentials gor GCS
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.gcs_credentials
logger.info(
    "🗝 Using GCS Credentials (for uploading model checkpoint): {}".format(
        args.gcs_credentials
    )
)

# Create wandb logger
wandb_logger = None
if args.wandb:
    logger.info("🪄Setting up wandb")
    wandb_logger = WandbLogger(project="SciBertPatentRanker")


# 🤖 Create Model
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(
    args.encoder_base,
)

# Form psql_args:
psql_args = {
    "host": args.psql_host,
    "database": args.psql_database,
    "port": args.psql_port,
    "user": args.psql_user,
    "password": args.psql_password,
    "relationships_table": args.psql_relationships_tablename,
    "samples_text": args.psql_reltexts_tablename,
}


###################################
# Create Training Environment
###################################
trainer = L.Trainer(
    accelerator="gpu",
    devices=1,
    logger=wandb_logger,
    accumulate_grad_batches=4,
    log_every_n_steps=1,
    max_epochs=args.epochs,
    # val_check_interval=0.1,
    enable_checkpointing=True,
)
logger.info("Passing the training to ⚡Pytorch-Lightning")
# Load Pytorch Lightning Modules
# Check if checkpoint exists
if Path("checkpoints/model.ckpt").exists():
    logger.info("Loading checkpoint")
    model = LightningMod_DocEmbedder.load_from_checkpoint(
        "checkpoints/model.ckpt",
        lang_head_name=args.encoder_base,
        tokenizer=tokenizer,
        load_init_weights=True,
    )
else:
    model = LightningMod_DocEmbedder(
        args.encoder_base, load_init_weights=True, tokenizer=tokenizer
    )
logger.info("🤖 Model Created")
###################################
# Prepare Data
###################################
# Create DataModule
dataModule = DataModule(
    args.batch_size,
    psql_args,
    args.data_path,
    tokenizer,
    args.encoder_max_length,
    device=device,
)
# dataModule.prepare_data()
tuner = Tuner(trainer)
tuner.scale_batch_size(model, mode="binsearch", datamodule=dataModule)
###################################
# Train
###################################
trainer.fit(model, datamodule=dataModule)
# Trainer save weigths
trainer.save_checkpoint("checkpoints/model.ckpt")
logger.info("⏫ Uploading Trained model to CGS")
upload_checkpoint_to_gcs(model, args.gcs_bucketname, args.gcs_blobname)
