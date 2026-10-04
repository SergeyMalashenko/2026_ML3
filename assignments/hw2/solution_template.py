"""Copy to solution.py and implement the functions. Do not install packages on import."""
import torch
from hw2_utils import select_device

DEVICE = select_device()


def get_dataloader(path, kind):
    """Return a DataLoader for train, val or test under path.

    The prepared layout supports torchvision.datasets.ImageFolder for all splits.
    Train and val use the same sorted 200 class names / label indices.
    Test has dummy labels (its only directory is images); never treat them as GT.
    Use deterministic transforms for val/test, shuffle=False and drop_last=False.
    Preserve test dataset.imgs order; do not reorder samples in collate_fn.
    Input tensors: float32, RGB, spatial dimensions at most 64 x 64.
    """
    raise NotImplementedError('Implement get_dataloader in solution.py')


def get_model():
    """Return your torch.nn.Module on DEVICE; initialize from scratch, no pretrained weights."""
    raise NotImplementedError('Implement get_model in solution.py')


def get_optimizer(model):
    """Return the optimizer used by train_on_tinyimagenet."""
    raise NotImplementedError('Implement get_optimizer in solution.py')


def predict(model, batch):
    """Return scores/logits of shape (batch_size, 200) in train class-index order.

    Move the input to the model device. Do not shuffle, drop or reorder samples.
    No softmax is required for argmax; for CrossEntropyLoss use raw logits.
    """
    raise NotImplementedError('Implement predict in solution.py')


def validate(dataloader, model):
    """Return (accuracy_fraction, mean_loss) over every validation example.

    Use eval and no_grad; weight batch losses by batch size before averaging.
    Restore the previous training mode if training continues afterwards.
    """
    raise NotImplementedError('Implement validate in solution.py')


def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer):
    """Train using train only, log train/val metrics and save your chosen checkpoint.

    Record configuration and seeds; validation is for model selection, not updates.
    Support resuming training with optimizer state in an additional checkpoint.
    """
    raise NotImplementedError('Implement train_on_tinyimagenet in solution.py')


def load_weights(model, checkpoint_path):
    """Load state_dict weights into model; respect its device (map_location).

    Prefer torch.load(..., weights_only=True). An ensemble may store several
    state_dicts in one checkpoint; document the format and load all of them here.
    Never require training to load the submitted model for inference.
    """
    raise NotImplementedError('Implement load_weights in solution.py')


def get_checkpoint_metadata():
    """Return (md5_checksum, Google_Drive_view_link) for the final checkpoint.

    Compute MD5 with hw2_utils.file_checksum('checkpoint.pth'), on any OS.
    An existing local file with the same checksum is accepted without Drive.
    Keep the link accessible to the reviewer, but do not distribute classmates' weights.
    The checksum identifies submitted bytes; it is not a security signature.
    """
    raise NotImplementedError('Set checksum and link in solution.py')
