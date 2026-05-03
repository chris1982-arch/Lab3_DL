# Image Captioning — Lab 3 Practical

CNN encoder + LSTM decoder with **concatenation fusion**, trained on the
Flickr8k dataset.

This project is split across three Python modules so that each team
member can work on a clearly defined part:

| Module          | Owner     | Status        | Responsibility |
|-----------------|-----------|---------------|----------------|
| `data.py`       | Person 1  | ✅ Done       | Dataset download, vocabulary, image-level split, DataLoaders |
| `model.py` + `train.py` | Person 2  | 🔲 In progress | CNN encoder, LSTM decoder, training loop, epoch validation |
| `evaluate.py` + `inference.py` | Person 3 | 🔲 Pending | BLEU score, sample captions, beam search (bonus) |

## Assignment requirements

The lab specifies four extra requirements on top of a basic captioning
model. The owner of each requirement is shown in brackets:

1. **Image-level train/val/test split** — no image appears in more than
   one split, even though each image has ~5 captions. *[Person 1 ✅]*
2. **Per-epoch validation function** — validation loss computed at the
   end of every training epoch. *[Person 2]*
3. **BLEU score on the test set** — BLEU-1, BLEU-2, BLEU-3, BLEU-4
   averaged across all test images, using all available reference
   captions per image. *[Person 3]*
4. **Sample captions** — show some test images alongside their
   ground-truth and the model's generated caption. *[Person 3]*

**Bonus:** Beam search decoding (used as tie-breaker in grading).
*[Person 3]*

## Dataset

We use **Flickr8k** (~8,000 images, 5 captions each) instead of MSCOCO
because the assignment explicitly allows a smaller dataset and Flickr8k
is the one used in the reference video tutorial. The reference Show,
Attend and Tell tutorial also recommends Flickr8k for limited compute.

The dataset is downloaded automatically from HuggingFace
(`jxie/flickr8k`); no Kaggle account required.

## Architecture

- **Encoder:** ResNet-50 pretrained on ImageNet, with the final
  classification head removed. Outputs a 2048-dim image feature vector.
- **Decoder:** LSTM with word embeddings of size 256 and hidden state
  of size 512.
- **Fusion:** *Concatenation* of the image embedding and the previous
  word embedding at every timestep, projected to the LSTM input size.
- **Output head:** Linear layer mapping the LSTM hidden state to
  vocabulary logits, trained with `CrossEntropyLoss`.

## Setup

```bash
pip install -r requirements.txt
```

The first run downloads Flickr8k (~1 GB) into `flickr8k/`.

## Quick start

To verify the data pipeline works:

```bash
python data.py
```

This runs a smoke test that prints split sizes, vocabulary size, and
a sample batch.

To train the model (once `train.py` is added):

```bash
python train.py
```

To evaluate on the test set with BLEU (once `evaluate.py` is added):

```bash
python evaluate.py
```

## Public API for downstream modules

`train.py` and `evaluate.py` should consume the data layer through a
single function call:

```python
from data import get_loaders

train_loader, val_loader, test_loader, vocab = get_loaders(
    batch_size=32,
    freq_threshold=5,
    image_size=224,
    num_workers=2,
)

# Each batch yields:
#   imgs:     [B, 3, 224, 224]   float tensor (ImageNet-normalized)
#   captions: [B, max_len]       long  tensor (<SOS>...words...<EOS>, padded)
#   lengths:  [B]                long  tensor (true caption lengths)
```

Special token IDs are accessible via `vocab.stoi["<PAD>"]` etc., and the
size of the output vocabulary is `len(vocab)`.

## Group members

- Christos Palantzidis  *(Person 1 — data pipeline)*
- *<Person 2 name>*    *(model + training)*
- *<Person 3 name>*    *(evaluation + inference)*
