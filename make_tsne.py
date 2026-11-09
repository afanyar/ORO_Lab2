"""Крок 5. t-SNE ознак: як класи розділяються зі зростанням глибини.

Рисунки:
  tsne_depths_<мережа>.png   панелі за глибинами; колір = справжній клас; у заголовку silhouette score
  tsne_errors_<мережа>.png   найглибший шар: помилково класифіковані точки позначено чорним хрестиком

Береться test-спліт. Ознаки спершу стандартизуються і стискаються PCA до 50 вимірів
(стандартна практика перед t-SNE: швидше і менше шуму).

Приклади:
    python make_tsne.py
    python make_tsne.py --model vgg16
    python make_tsne.py --depths conv1 conv3 conv5 fc7
    python make_tsne.py --classes cat dog horse deer      # лише «тварини, що плутаються»
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def embed(X, perplexity, seed):
    """Стандартизація -> PCA(<=50) -> t-SNE у 2D. Повертає (2D-вкладення, PCA-ознаки)."""
    X = StandardScaler().fit_transform(X)
    n_comp = min(50, X.shape[1], X.shape[0] - 1)
    Xp = PCA(n_components=n_comp, random_state=seed).fit_transform(X)
    Z = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(Xp)
    return Z, Xp


def palette(n):
    return plt.get_cmap("tab10" if n <= 10 else "tab20")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", default="features")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--dataset", default="stl10")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--model", default=None, help="мережа (за замовчуванням найкраща за val)")
    p.add_argument("--depths", nargs="*", default=None, help="глибини для панелей (за замовчуванням 4 рівномірно)")
    p.add_argument("--classes", nargs="*", default=None, help="показати лише ці класи")
    p.add_argument("--perplexity", type=float, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(args.results_dir) / "results.csv")
    pr = np.load(Path(args.results_dir) / "predictions.npz")

    model = args.model or df.loc[df.val_acc.idxmax()].model
    data = np.load(Path(args.features_dir) / f"{args.dataset}_{model}.npz")
    all_depths = [str(d) for d in data["depths"]]
    classes = [str(c) for c in data["classes"]]
    y = data["test__labels"]

    depths = args.depths or [all_depths[i] for i in sorted(set(np.linspace(0, len(all_depths) - 1, 4).round().astype(int)))]
    bad = [d for d in depths if d not in all_depths]
    if bad:
        raise SystemExit(f"Невідомі глибини {bad}. Доступні: {all_depths}")

    # маска зображень (для --classes)
    if args.classes:
        unknown = set(args.classes) - set(classes)
        if unknown:
            raise SystemExit(f"Невідомі класи: {sorted(unknown)}. Доступні: {classes}")
        keep_ids = [classes.index(c) for c in args.classes]
        mask = np.isin(y, keep_ids)
    else:
        keep_ids = list(range(len(classes)))
        mask = np.ones(len(y), dtype=bool)
    y_m = y[mask]
    cmap = palette(len(classes))

    def scatter(ax, Z, labels, s=6):
        for cid in keep_ids:
            sel = labels == cid
            ax.scatter(Z[sel, 0], Z[sel, 1], s=s, alpha=0.7, color=cmap(cid), label=classes[cid])
        ax.set_xticks([])
        ax.set_yticks([])

    # --- Рисунок 1: t-SNE за глибинами --------------------------------------------
    fig, axes = plt.subplots(1, len(depths), figsize=(4.6 * len(depths), 4.6), squeeze=False)
    for ax, d in zip(axes[0], depths):
        print(f"t-SNE: {model} / {d} ...", flush=True)
        Z, Xp = embed(data[f"test__{d}"][mask], args.perplexity, args.seed)
        sil = silhouette_score(Xp, y_m)
        scatter(ax, Z, y_m)
        ax.set_title(f"{model} / {d}\nsilhouette = {sil:.2f}", fontsize=10)
    axes[0][-1].legend(markerscale=2.5, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(out / f"tsne_depths_{model}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Рисунок 2: найглибший шар з позначеними помилками --------------------------
    last = all_depths[-1]
    row = df[(df.model == model) & (df.depth == last)]
    row = row.loc[row.val_acc.idxmax()]
    pred = pr[f"{model}__{last}__{row.clf}"]
    print(f"t-SNE з помилками: {model} / {last} / {row.clf} ...", flush=True)
    Z, _ = embed(data[f"test__{last}"][mask], args.perplexity, args.seed)
    wrong = pred[mask] != y_m
    fig, ax = plt.subplots(figsize=(7.5, 6))
    scatter(ax, Z, y_m)
    ax.scatter(Z[wrong, 0], Z[wrong, 1], s=28, marker="x", color="black", linewidths=1.0,
               label="помилка класифікації")
    ax.set_title(f"{model} / {last} / {row.clf}: помилок {wrong.sum()} з {len(y_m)}", fontsize=10)
    ax.legend(markerscale=2, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(out / f"tsne_errors_{model}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Рисунки збережено в {out.resolve()}")


if __name__ == "__main__":
    main()
