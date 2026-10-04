"""Infrastructure for HW2, not a model or training solution."""
from decimal import Decimal, ROUND_CEILING
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

DATASET_URL = 'https://cs231n.stanford.edu/tiny-imagenet-200.zip'


def select_device(force_cpu=False):
    import torch
    if force_cpu:
        return torch.device('cpu')
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def file_checksum(path, algorithm='md5'):
    digest = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(parent='data', url=DATASET_URL):
    """Download the original dataset and arrange val for ImageFolder, idempotently."""
    parent = Path(parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / 'tiny-imagenet-200'
    marker = root / '.hw2-prepared'
    if marker.is_file():
        validate_dataset(root)
        return root
    # A failed extraction never replaces a previously usable dataset.
    archive = parent / 'tiny-imagenet-200.zip'
    if not archive.is_file():
        part = archive.with_suffix('.zip.part')
        try:
            if shutil.which('curl'):
                subprocess.run(['curl', '--fail', '--location', '--retry', '2',
                                '--proto', '=https', '--proto-redir', '=https',
                                '--output', str(part), url], check=True, timeout=1200)
            else:
                with urlopen(url, timeout=60) as response, part.open('wb') as output:
                    shutil.copyfileobj(response, output)
            with ZipFile(part) as zipped:
                if zipped.testzip() is not None:
                    raise ValueError('Dataset ZIP failed its CRC check.')
            part.replace(archive)
        finally:
            part.unlink(missing_ok=True)
    if not root.is_dir():
        with tempfile.TemporaryDirectory(dir=parent, prefix='.extract-') as temporary:
            destination = Path(temporary).resolve()
            with ZipFile(archive) as zipped:
                for info in zipped.infolist():
                    if not (destination / info.filename).resolve().is_relative_to(destination):
                        raise ValueError('Unsafe path in dataset ZIP.')
                zipped.extractall(destination)
            extracted = destination / root.name
            if not extracted.is_dir():
                raise ValueError('ZIP does not contain tiny-imagenet-200.')
            extracted.rename(root)
    annotations = root / 'val' / 'val_annotations.txt'
    if annotations.is_file():
        for line in annotations.read_text().splitlines():
            name, label, *_ = line.split()
            source = root / 'val' / 'images' / name
            target = root / 'val' / label / name
            if source.is_file():
                target.parent.mkdir(exist_ok=True)
                if target.is_file() and file_checksum(source) != file_checksum(target):
                    raise ValueError(f'Conflicting validation image: {name}')
                source.replace(target)
            elif not target.is_file():
                raise FileNotFoundError(source)
        image_directory = root / 'val' / 'images'
        if image_directory.is_dir() and not any(image_directory.iterdir()):
            image_directory.rmdir()
    validate_dataset(root)
    marker.write_text('HW2 dataset prepared; validation annotations preserved.\n')
    return root


def validate_dataset(root):
    root = Path(root)
    classes = sorted(line.strip() for line in (root / 'wnids.txt').read_text().splitlines() if line.strip())
    if len(classes) != 200 or len(set(classes)) != 200:
        raise ValueError('Expected 200 distinct classes in wnids.txt.')
    for split, expected in [('train', 100000), ('val', 10000), ('test', 10000)]:
        actual = list((root / split).rglob('*.JPEG'))
        if len(actual) != expected:
            raise ValueError(f'{split}: expected {expected} images, found {len(actual)}.')
    train_classes = sorted(p.name for p in (root / 'train').iterdir() if p.is_dir())
    if train_classes != classes:
        raise ValueError('Training classes do not match wnids.txt.')
    expected_ids = {f'test_{i}.JPEG' for i in range(10000)}
    if {p.name for p in (root / 'test').rglob('*.JPEG')} != expected_ids:
        raise ValueError('Unexpected test image IDs; compare with the Kaggle dataset.')
    return classes


def prepare_checkpoint(solution, destination='checkpoint.pth'):
    """Use an existing checkpoint or download it, and fail closed on hash mismatch."""
    expected, url = solution.get_checkpoint_metadata()
    expected = str(expected).strip().lower()
    if len(expected) != 32 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('get_checkpoint_metadata() must return a real MD5 checksum.')
    destination = Path(destination).expanduser()
    if destination.is_file():
        if file_checksum(destination) != expected:
            raise ValueError('Local checkpoint checksum differs. Select the correct weights file.')
        return destination
    if not isinstance(url, str) or not url.startswith('https://drive.google.com/'):
        raise ValueError('Upload weights locally, or provide an HTTPS Google Drive view link.')
    import gdown
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + '.part')
    try:
        result = gdown.download(url=url, output=str(part), fuzzy=True, quiet=False)
        if result is None or not part.is_file():
            raise RuntimeError('Weights download failed. Check access or upload the exact file locally.')
        if file_checksum(part) != expected:
            raise ValueError('Downloaded checkpoint checksum differs; weights were not loaded.')
        part.replace(destination)
    finally:
        part.unlink(missing_ok=True)
    return destination


def make_submission(solution, model, test_loader, classes, output='submission.csv', sample_path=None):
    """Reject loaders whose iteration order cannot be matched to dataset.imgs."""
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import BatchSampler, DataLoader, SequentialSampler
    if not isinstance(test_loader, DataLoader):
        raise TypeError('Use a torch DataLoader for the checked test inference.')
    if type(test_loader.sampler) is not SequentialSampler or test_loader.drop_last:
        raise ValueError('Test loader requires shuffle=False and drop_last=False.')
    if type(test_loader.batch_sampler) is not BatchSampler:
        raise ValueError('Use the standard sequential batch sampler for test inference.')
    images = getattr(test_loader.dataset, 'imgs', None)
    if images is None or len(images) != len(test_loader.dataset):
        raise ValueError('Test dataset must expose ordered .imgs, as torchvision ImageFolder does.')
    ids = [Path(item[0]).name for item in images]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate test image IDs.')
    if len(classes) != 200 or len(set(classes)) != 200:
        raise ValueError('Pass the 200 training dataset classes in label-index order.')
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch, _ in test_loader:
            scores = solution.predict(model, batch)
            if not isinstance(scores, torch.Tensor) or tuple(scores.shape) != (len(batch), len(classes)):
                raise ValueError('predict must return a tensor of shape (batch_size, 200).')
            if not torch.isfinite(scores).all():
                raise ValueError('Non-finite model predictions.')
            predictions.extend(scores.argmax(dim=1).detach().cpu().tolist())
    if len(predictions) != len(ids):
        raise ValueError('Number of predictions differs from number of test images.')
    frame = pd.DataFrame({'id': ids, 'pred': np.asarray(classes)[predictions]})
    if sample_path is not None:
        sample = pd.read_csv(sample_path, dtype=str)
        if list(sample.columns) != ['id', 'pred'] or sample.id.duplicated().any():
            raise ValueError('Unexpected sample_submission schema.')
        if set(sample.id) != set(frame.id):
            raise ValueError('Test image IDs differ from sample_submission.csv.')
        frame = frame.set_index('id').loc[sample.id].reset_index()
    frame.to_csv(output, index=False)
    return frame


def grade_hw2(public_accuracy, private_accuracy, report=3, logging=3, private_place=None):
    """Arithmetic only; reproducibility and compliance are checked separately."""
    public, private = Decimal(str(public_accuracy)), Decimal(str(private_accuracy))
    report, logging = Decimal(str(report)), Decimal(str(logging))
    if any(not x.is_finite() for x in (public, private, report, logging)):
        raise ValueError('Arguments must be finite.')
    if not (0 <= public <= 1 and 0 <= private <= 1):
        raise ValueError('Accuracy must be a fraction from 0 to 1.')
    if not (0 <= report <= 3 and 0 <= logging <= 3):
        raise ValueError('Report and logging each range from 0 to 3.')
    if private_place is not None and (type(private_place) is not int or private_place < 1):
        raise ValueError('Private place must be a positive integer or None.')
    quality = min(Decimal(14), 28 * public) if public >= Decimal('.25') else Decimal(0)
    bonus = max(Decimal(0), 28 * private - 14)
    podium = {1: 5, 2: 3, 3: 1}.get(private_place, 0)
    return min(30, int((report + logging + quality + bonus + podium).to_integral_value(rounding=ROUND_CEILING)))
