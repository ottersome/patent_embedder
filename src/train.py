import logging
import os
from pathlib import Path, PosixPath

import lightning as L
import torch.utils.data as data
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertConfig

from .dataset import DatabaseFactory
from .lightningMods import LightningMod_DocEmbedder
from .utils.general import getargs, upload_checkpoint_to_gcs

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
logger.info("🗝 Using GCS Credentials: {}".format(args.gcs_credentials))

# Create wandb logger
wandb_logger = None
if args.wandb:
    logger.info("🪄Setting up wandb")
    wandb_logger = WandbLogger(project="SciBertPatentRanker")


# 🤖 Create Model
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
# Prepare Data
###################################
# Create Dataset
logger.info("📂Creating Dataset")
datasetFactory = DatabaseFactory(
    dir_to_cache=PosixPath(args.data_path),
    psql_args=psql_args,
    tokenizer=tokenizer,
    encoder_max_length=args.encoder_max_length,
)
train_dataset, val_dataset = datasetFactory.createDataset()

logger.info("Passing the traiing to ⚡Pytorch-Lightning")
# Load Pytorch Lightning Modules
# Check if checkpoint exists
if Path("checkpoints/model.ckpt").exists():
    logger.info("Loading checkpoint")
    lightning_modules = LightningMod_DocEmbedder.load_from_checkpoint(
        "checkpoints/model.ckpt",
        lang_head_name=args.encoder_base,
        load_init_weights=False,
    )
else:
    lightning_modules = LightningMod_DocEmbedder(
        args.encoder_base, load_init_weights=True
    )
logger.info("🤖Model Created")

trainer = L.Trainer(
    logger=wandb_logger,
    accumulate_grad_batches=4,
    max_epochs=args.epochs,
    val_check_interval=0.25,
    enable_checkpointing=True,
)
# tuner = Tuner(trainer)
# tuner.scale_batch_size(lightning_modules, mode="binsearch")
trainer.fit(
    lightning_modules,
    data.DataLoader(train_dataset, batch_size=2, num_workers=12),
    data.DataLoader(val_dataset, batch_size=2, num_workers=12),
)
# Trainer save weigths
trainer.save_checkpoint("checkpoints/model.ckpt")
logger.info("⏫ Uploading Trained model to CGS")
upload_checkpoint_to_gcs(lightning_modules, args.gcs_bucketname, args.gcs_blobname)
