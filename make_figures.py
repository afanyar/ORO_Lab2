"""Крок 4. Будує всі рисунки для звіту з результатів run_experiments.py.

  trend_accuracy.png          точність залежно від глибини (панель на кожну мережу, лінія на класифікатор)
  cm_grid_best_depth.png      confusion matrices: мережа x класифікатор, на найкращій (за val) глибині
  cm/<мережа>_<clf>.png       ті самі матриці окремими файлами
  cm_depth_<мережа>_<clf>.png еволюція матриці зі зростанням глибини
  typical_errors.png          приклади найтиповіших помилкових класифікацій
  top_confusions.csv          пари класів, які плутаються найчастіше

Приклади:
    python make_figures.py
    python make_figures.py --model vgg16 --clf SVM-rbf
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix


def draw_cm(ax, cm, classes, title):
    n = len(classes)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(classes, fontsize=7)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > 50 else "black")
    ax.set_xlabel("Передбачений клас", fontsize=8)
    ax.set_ylabel("Справжній клас", fontsize=8)
    ax.set_title(title, fontsize=9)


def norm_cm(y_true, pred, n):
    return confusion_matrix(y_true, pred, labels=range(n), normalize="true") * 100


def best_row(df, model=None, clf=None):
    d = df
    if model:
        d = d[d.model == model]
    if clf:
        d = d[d.clf == clf]
    if d.empty:
        raise SystemExit(f"Немає результатів для model={model}, clf={clf}")
    return d.loc[d.val_acc.idxmax()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--model", default=None, help="мережа для рисунків про глибину та помилки (за замовчуванням найкраща)")
    p.add_argument("--clf", default=None, help="класифікатор для тих самих рисунків (за замовчуванням найкращий)")
    p.add_argument("--top-pairs", type=int, default=3, help="скільки пар класів показати у typical_errors")
    p.add_argument("--n-examples", type=int, default=6, help="скільки прикладів на пару")
    args = p.parse_args()

    res = Path(args.results_dir)
    out = Path(args.out_dir)
    (out / "cm").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(res / "results.csv")
    pr = np.load(res / "predictions.npz")
    classes = [str(c) for c in pr["classes"]]
    y_te = pr["y_test"]
    paths = pr["test_paths"]
    n_cls = len(classes)
    n_test = len(y_te)

    def pred_of(r):
        return pr[f"{r.model}__{r.depth}__{r.clf}"]

    models = list(dict.fromkeys(df.model))
    clfs = list(dict.fromkeys(df.clf))

    # --- 1. Тренд точності ------------------------------------------------------
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.2), squeeze=False, sharey=True)
    for ax, m in zip(axes[0], models):
        sub = df[df.model == m]
        depths = list(dict.fromkeys(sub.depth))
        for c in clfs:
            s = sub[sub.clf == c].set_index("depth").reindex(depths)  # пропуски, якщо глибину ще не пораховано
            acc = s.test_acc.values
            err = 1.96 * np.sqrt(acc * (1 - acc) / n_test)  # 95% інтервал для частки
            ax.errorbar(depths, acc, yerr=err, marker="o", capsize=3, label=c)
        ax.set_title(m)
        ax.set_xlabel("Глибина (шар, з якого взято ознаки)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Test accuracy")
    axes[0][-1].legend(title="Класифікатор", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "trend_accuracy.png", dpi=150)
    plt.close(fig)

    # --- 2. Confusion matrices: мережа x класифікатор на найкращій глибині --------
    fig, axes = plt.subplots(len(models), len(clfs), figsize=(4.8 * len(clfs), 4.8 * len(models)), squeeze=False)
    for i, m in enumerate(models):
        for j, c in enumerate(clfs):
            r = best_row(df, m, c)
            cm = norm_cm(y_te, pred_of(r), n_cls)
            title = f"{m} / {c}\nглибина {r.depth}, acc {r.test_acc:.3f}"
            draw_cm(axes[i][j], cm, classes, title)
            f1, a1 = plt.subplots(figsize=(5.5, 5.5))
            draw_cm(a1, cm, classes, title)
            f1.tight_layout()
            f1.savefig(out / "cm" / f"{m}_{c}.png", dpi=150)
            plt.close(f1)
    fig.tight_layout()
    fig.savefig(out / "cm_grid_best_depth.png", dpi=130)
    plt.close(fig)

    # --- 3. Еволюція матриці з глибиною для обраної пари (мережа, класифікатор) ---
    chosen = best_row(df, args.model, args.clf)
    sub = df[(df.model == chosen.model) & (df.clf == chosen.clf)]
    cols = 4
    nrows = int(np.ceil(len(sub) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(4.6 * cols, 4.6 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, sub.iterrows()):
        ax.axis("on")
        draw_cm(ax, norm_cm(y_te, pred_of(r), n_cls), classes, f"{r.depth}, acc {r.test_acc:.3f}")
    fig.suptitle(f"{chosen.model} / {chosen.clf}: матриця плутанини залежно від глибини")
    fig.tight_layout()
    fig.savefig(out / f"cm_depth_{chosen.model}_{chosen.clf}.png", dpi=130)
    plt.close(fig)

    # --- 4. Типові помилки для найкращої конфігурації ------------------------------
    pred = pred_of(chosen)
    cm_cnt = confusion_matrix(y_te, pred, labels=range(n_cls))
    pairs = [(cm_cnt[t, q], t, q) for t in range(n_cls) for q in range(n_cls) if t != q and cm_cnt[t, q] > 0]
    pairs.sort(reverse=True)
    pd.DataFrame(
        [{"true": classes[t], "predicted": classes[q], "count": int(c),
          "share_of_true_class_%": round(100 * c / (y_te == t).sum(), 1)} for c, t, q in pairs]
    ).to_csv(out / "top_confusions.csv", index=False)

    top = pairs[: args.top_pairs]
    if top:
        fig, axes = plt.subplots(len(top), args.n_examples, figsize=(2.2 * args.n_examples, 2.5 * len(top)), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for row, (c, t, q) in enumerate(top):
            idx = np.where((y_te == t) & (pred == q))[0][: args.n_examples]
            for col, k in enumerate(idx):
                try:
                    axes[row][col].imshow(Image.open(str(paths[k])).convert("RGB"))
                except FileNotFoundError:
                    axes[row][col].text(0.5, 0.5, "файл\nне знайдено", ha="center", va="center")
                axes[row][col].set_title(f"{classes[t]} → {classes[q]}" if col == 0 else "", fontsize=9, loc="left")
            axes[row][0].set_ylabel(f"n={c}")
        fig.suptitle(f"Типові помилки: {chosen.model} / {chosen.depth} / {chosen.clf}")
        fig.tight_layout()
        fig.savefig(out / "typical_errors.png", dpi=130)
        plt.close(fig)

    print(f"Обрана конфігурація для рисунків 3-4: {chosen.model} / {chosen.depth} / {chosen.clf} "
          f"(val={chosen.val_acc:.3f}, test={chosen.test_acc:.3f})")
    print("Топ плутанин:")
    for c, t, q in pairs[:10]:
        print(f"  {classes[t]} → {classes[q]}: {c}")
    print(f"Рисунки збережено в {out.resolve()}")


if __name__ == "__main__":
    main()
