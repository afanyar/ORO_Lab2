"""Крок 1. Готує датасет у форматі ImageFolder:

    data/stl10/{train,val,test}/<клас>/*.png

Джерела STL-10:
  --source hf           (за замовчуванням) Hugging Face, датасет tanganke/stl10, ~230 МБ
  --source torchvision  оригінал зі Stanford (~2.5 ГБ); сервер інколи недоступний

Якщо береш власний датасет — просто розклади його в таку саму структуру
папок і пропусти цей скрипт.

Приклади:
    python prepare_dataset.py
    python prepare_dataset.py --classes airplane car cat dog horse ship
    python prepare_dataset.py --n-train 200 --n-val 50 --n-test 150
"""
import argparse
import random
from pathlib import Path

import numpy as np

# Офіційний порядок класів STL-10 (запасний варіант, якщо в датасеті немає назв)
STL10_CLASSES = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]


class Split:
    """Мінімальна обгортка: масив міток + функція, що повертає PIL-зображення за індексом."""

    def __init__(self, labels, get_image):
        self.labels = np.asarray(labels)
        self.get_image = get_image


def load_hf(cache_dir):
    from datasets import load_dataset  # бібліотека Hugging Face (у Colab вже встановлена)

    data = load_dataset("tanganke/stl10", cache_dir=cache_dir)
    train, test = data["train"], data["test"]
    feats = train.features
    img_col = next(k for k, f in feats.items() if type(f).__name__ == "Image")
    lab_col = next(k for k, f in feats.items() if type(f).__name__ == "ClassLabel")
    classes = list(feats[lab_col].names)
    if all(str(c).isdigit() for c in classes):  # назв немає — беремо офіційний порядок
        classes = STL10_CLASSES

    def wrap(ds):
        return Split(ds[lab_col], lambda i: ds[int(i)][img_col])

    return wrap(train), wrap(test), classes


def load_torchvision(root):
    from torchvision import datasets as tv_datasets

    train = tv_datasets.STL10(root, split="train", download=True)
    test = tv_datasets.STL10(root, split="test", download=True)

    def wrap(ds):
        return Split(ds.labels, lambda i: ds[int(i)][0])

    return wrap(train), wrap(test), list(train.classes)


def save_images(split, indices, split_dir, class_name):
    class_dir = split_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for i in indices:
        split.get_image(i).convert("RGB").save(class_dir / f"{int(i):05d}.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["hf", "torchvision"], default="hf")
    p.add_argument("--root", default="data/raw", help="куди качати/кешувати оригінальний датасет")
    p.add_argument("--out", default="data/stl10", help="куди зберегти ImageFolder")
    p.add_argument("--classes", nargs="*", default=None, help="підмножина класів (за замовчуванням усі 10)")
    p.add_argument("--n-train", type=int, default=350, help="зображень на клас у train")
    p.add_argument("--n-val", type=int, default=100, help="зображень на клас у val")
    p.add_argument("--n-test", type=int, default=300, help="зображень на клас у test")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    if args.source == "hf":
        train_ds, test_ds, classes = load_hf(args.root)
    else:
        train_ds, test_ds, classes = load_torchvision(args.root)
    print(f"Класи: {classes}")

    wanted = args.classes or classes
    unknown = set(wanted) - set(classes)
    if unknown:
        raise SystemExit(f"Невідомі класи: {sorted(unknown)}. Доступні: {classes}")

    out = Path(args.out)
    for name in wanted:
        label = classes.index(name)
        tr = np.where(train_ds.labels == label)[0].tolist()
        te = np.where(test_ds.labels == label)[0].tolist()
        if len(tr) < args.n_train + args.n_val:
            raise SystemExit(f"{name}: у train лише {len(tr)} зображень, а потрібно {args.n_train + args.n_val}")
        if len(te) < args.n_test:
            raise SystemExit(f"{name}: у test лише {len(te)} зображень, а потрібно {args.n_test}")
        rng.shuffle(tr)
        rng.shuffle(te)
        save_images(train_ds, tr[: args.n_train], out / "train", name)
        save_images(train_ds, tr[args.n_train : args.n_train + args.n_val], out / "val", name)
        save_images(test_ds, te[: args.n_test], out / "test", name)
        print(f"{name}: train={args.n_train}, val={args.n_val}, test={args.n_test}")

    print(f"Готово: {out.resolve()}")


if __name__ == "__main__":
    main()
