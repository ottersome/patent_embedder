import os
from argparse import ArgumentParser
from pathlib import Path

from transformers import AutoTokenizer

from docembedder import DocEmbedder, DocRep
from src.lightningMods import LightningMod_DocEmbedder

# Key location
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcs_service_key.json"
ap = ArgumentParser()
ap.add_argument("--path", type=str, default="checkpoints/model.ckpt")
args = ap.parse_args()

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

# Original file, related file, and non-related file
og_file_title = "USB connection-detection circuitry and operation methods of the same"
og_file_abstract = "A USB connection-detection circuitry and the operation method of the same are disclosed. The circuitry includes a transmitting circuit and a detecting circuit. The transmitting circuit contains a pair of differential signal lines, a pair of pull-down resistors and a pair of pull-up resistors wherein one pull-down resistor and one pull-up resistor are connected to the same differential signal line with their own individidual"

related_file_title = "Microcontroller input/output nodes with both programmable pull-up and pull-down resistive loads and programmable drive strength "
related_file_abstract = "The present invention relates to an input/output node in an electronic device which comprises an input/output pin, a plurality of programmable pull-up resistors and a plurality of programmable pull-down resistors. Each of the pull-up and pull-down resistors, or a combination of them, can be activated by turning on or off n-MOS and p-MOS transistors with logic c"

non_related_file_title = "Reusable pesticide bait station"
non_related_file_abstract = "A compact reusable bait station that can be used for either solid or liquid bait. The bait station has an outer wall that defines an inner bait chamber. The chamber, when sealed, prevents liquid bait inside the chamber from flowing out. The chamber has limited access from the outside to reduce evaporation, drying, and contamination of the liquid bait by preventing unnecessary exposure to the environment. Access to the outside is limited to access ports th"

# Create DocRep objects
og_doc = DocRep(og_file_title, og_file_abstract)
related_doc = DocRep(related_file_title, related_file_abstract)
non_related_doc = DocRep(non_related_file_title, non_related_file_abstract)

# Create DocEmbedder object
if Path(args.path).exists():
    doc_embedded = DocEmbedder(args.path, tokenizer)

    print(
        "Distance between original and related:",
        doc_embedded.doc_dist(og_doc, related_doc),
    )
    print(
        "Distance between original and non-related:",
        doc_embedded.doc_dist(og_doc, non_related_doc),
    )
else:
    raise ValueError("Invalid path")

# Get distances between documents

# Thats pretty much it
