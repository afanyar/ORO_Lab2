"""Крок 2. Витягує ознаки з різної глибини попередньо навчених мереж (ImageNet)
і зберігає їх у features/<датасет>_<мережа>.npz.

Ідея (варіант А): мережа повністю заморожена, ми беремо виходи проміжних шарів
(кожен -> вектор через global average pooling) і далі навчаємо поверх них
класифікатори. Чим глибший шар, тим «вища» ознака.

Приклади:
    python extract_features.py --data-dir data/stl10
    python extract_features.py --data-dir data/stl10 --models vgg16 --batch-size 32
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ---- точки «глибини»: (назва, модуль, з якого знімаємо вихід) ----------------
def alexnet_points(m):
    return [
        ("conv1", m.features[1]),    # ReLU після conv1
        ("conv2", m.features[4]),
        ("conv3", m.features[7]),
        ("conv4", m.features[9]),
        ("conv5", m.features[11]),
        ("fc6", m.classifier[2]),    # ReLU після fc6
        ("fc7", m.classifier[5]),    # ReLU після fc7
    ]


def vgg16_points(m):
    return [
        ("block1", m.features[4]),   # MaxPool в кінці кожного блоку
        ("block2", m.features[9]),
        ("block3", m.features[16]),
        ("block4", m.features[23]),
        ("block5", m.features[30]),
        ("fc6", m.classifier[1]),
        ("fc7", m.classifier[4]),
    ]


def resnet18_points(m):
    return [
        ("stem", m.relu),
        ("layer1", m.layer1),
        ("layer2", m.layer2),
        ("layer3", m.layer3),
        ("layer4", m.layer4),
    ]


MODELS = {
    "alexnet": (models.alexnet, models.AlexNet_Weights.IMAGENET1K_V1, alexnet_points),
    "vgg16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1, vgg16_points),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, resnet18_points),
}


def make_hook(store, name, pool):
    def hook(module, inputs, output):
        x = output.detach()
        if x.ndim == 4:  # згорткові карти: (B, C, H, W) -> (B, C * pool * pool)
            x = F.adaptive_avg_pool2d(x, pool).flatten(1)
        else:            # повнозвʼязні шари: копіюємо, бо ReLU може бути inplace
            x = x.clone()
        store[name].append(x.cpu().numpy().astype(np.float32))
    return hook


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/stl10", help="папка з train/ val/ test/ (ImageFolder)")
    p.add_argument("--out-dir", default="features")
    p.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--pool", type=int, default=1,
                   help="розмір adaptive avg pool для згорткових шарів (1 = GAP)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # датасети створюємо один раз і перевіряємо, що класи збігаються в усіх спліттах
    splits = {}
    classes = None
    for split in ("train", "val", "test"):
        ds = datasets.ImageFolder(data_dir / split, tfm)
        if classes is None:
            classes = ds.classes
        elif ds.classes != classes:
            raise SystemExit(f"Набір класів у '{split}' відрізняється від 'train'")
        splits[split] = ds
        print(f"{split}: {len(ds)} зображень, {len(classes)} класів")

    for model_name in args.models:
        ctor, weights, points_fn = MODELS[model_name]
        model = ctor(weights=weights).to(device).eval()
        points = points_fn(model)
        print(f"\n=== {model_name}: глибини {[n for n, _ in points]}")

        result = {
            "classes": np.array(classes),
            "depths": np.array([n for n, _ in points]),
        }
        t0 = time.time()
        for split, ds in splits.items():
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers)
            store = {name: [] for name, _ in points}
            handles = [mod.register_forward_hook(make_hook(store, name, args.pool))
                       for name, mod in points]
            with torch.inference_mode():
                for i, (x, _) in enumerate(loader, 1):
                    model(x.to(device))
                    if i % 10 == 0 or i == len(loader):
                        print(f"  {split}: батч {i}/{len(loader)}  ({time.time() - t0:.0f} с)")
            for h in handles:
                h.remove()

            for name in store:
                result[f"{split}__{name}"] = np.concatenate(store[name])
            result[f"{split}__labels"] = np.array(ds.targets)
            result[f"{split}__paths"] = np.array([path for path, _ in ds.samples])

        out_file = out_dir / f"{data_dir.name}_{model_name}.npz"
        np.savez(out_file, **result)
        print(f"Збережено {out_file}")
        for name, _ in points:
            print(f"  {name}: train {result[f'train__{name}'].shape}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
