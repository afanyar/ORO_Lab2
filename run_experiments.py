"""Крок 3. Навчає класифікатори поверх ознак з кожної глибини кожної мережі.

Для кожної комбінації (мережа, глибина, класифікатор):
  1. навчає модель на train для кількох значень гіперпараметра,
  2. обирає найкраще значення за точністю на val,
  3. міряє точність на test (test для підбору не використовується).

Результати (зберігаються після кожної глибини, тож запуск можна перервати):
  results/results.csv       таблиця точностей
  results/predictions.npz   передбачення на test (для confusion matrices і аналізу помилок)

Повторний запуск ПРОДОВЖУЄ з місця зупинки: готові комбінації пропускаються.
Щоб почати з нуля, додай --fresh.

Приклади:
    python run_experiments.py
    python run_experiments.py --models vgg16          # лише одна мережа
    python run_experiments.py --classifiers SVM-linear RF   # швидше
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SHARED_KEYS = ("classes", "y_test", "test_paths")

# назва -> (функція, що будує класифікатор за значенням гіперпараметра, значення, чи потрібен StandardScaler)
CLASSIFIERS = {
    "SVM-linear": (lambda c: SVC(kernel="linear", C=c, cache_size=1000), [0.001, 0.01, 0.1], True),
    "SVM-rbf": (lambda c: SVC(kernel="rbf", C=c, gamma="scale", cache_size=1000), [1, 10], True),
    "LogReg": (lambda c: LogisticRegression(C=c, max_iter=300), [0.01, 0.1], True),
    "RF": (lambda n: RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=0), [200], False),
}


def build(name, param):
    make, _, scale = CLASSIFIERS[name]
    clf = make(param)
    return make_pipeline(StandardScaler(), clf) if scale else clf


def save_state(out_dir, rows, preds, shared):
    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)
    np.savez(out_dir / "predictions.npz", **shared, **preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", default="features")
    p.add_argument("--dataset", default="stl10", help="префікс файлів ознак: <dataset>_<мережа>.npz")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--models", nargs="*", default=None, help="за замовчуванням усі знайдені")
    p.add_argument("--classifiers", nargs="*", default=list(CLASSIFIERS), choices=list(CLASSIFIERS))
    p.add_argument("--fresh", action="store_true", help="ігнорувати попередні результати і почати з нуля")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.features_dir).glob(f"{args.dataset}_*.npz"))
    if not files:
        raise SystemExit(f"Не знайдено файлів {args.dataset}_*.npz у {args.features_dir}")

    rows, preds, shared = [], {}, {}
    csv_path, npz_path = out_dir / "results.csv", out_dir / "predictions.npz"
    if not args.fresh and csv_path.exists() and npz_path.exists():
        rows = pd.read_csv(csv_path).to_dict("records")
        old = np.load(npz_path)
        for k in old.files:
            (shared if k in SHARED_KEYS else preds)[k] = old[k]
        print(f"Знайдено {len(rows)} готових результатів — вони пропускаються (--fresh, щоб почати з нуля)")
    done = {(r["model"], str(r["depth"]), r["clf"]) for r in rows}

    t0 = time.time()
    for f in files:
        model_name = f.stem[len(args.dataset) + 1:]
        if args.models and model_name not in args.models:
            continue
        data = np.load(f)
        depths = [str(d) for d in data["depths"]]
        y_tr, y_va, y_te = data["train__labels"], data["val__labels"], data["test__labels"]
        if not shared:
            shared = {"classes": data["classes"], "y_test": y_te, "test_paths": data["test__paths"]}
        print(f"\n=== {model_name}: глибини {depths}")

        for depth in depths:
            todo = [c for c in args.classifiers if (model_name, depth, c) not in done]
            if not todo:
                continue
            X_tr, X_va, X_te = (data[f"{s}__{depth}"] for s in ("train", "val", "test"))
            for clf_name in todo:
                print(f"  {depth:8s} {clf_name:11s} ...", end="", flush=True)
                _, params, _ = CLASSIFIERS[clf_name]
                best = None
                for prm in params:
                    model = build(clf_name, prm)
                    model.fit(X_tr, y_tr)
                    val_acc = float(model.score(X_va, y_va))
                    if best is None or val_acc > best[0]:
                        best = (val_acc, prm, model)
                val_acc, prm, model = best
                pred = model.predict(X_te)
                test_acc = float((pred == y_te).mean())
                rows.append({
                    "model": model_name, "depth": depth, "clf": clf_name,
                    "param": prm, "val_acc": val_acc, "test_acc": test_acc,
                    "n_features": X_tr.shape[1],
                })
                preds[f"{model_name}__{depth}__{clf_name}"] = pred
                print(f" param={prm:<6} val={val_acc:.3f} test={test_acc:.3f}  ({time.time() - t0:.0f} с)")
            save_state(out_dir, rows, preds, shared)  # контрольна точка після кожної глибини

    if not rows:
        raise SystemExit("Немає результатів: перевір --models і --classifiers")
    save_state(out_dir, rows, preds, shared)
    df = pd.DataFrame(rows)

    print("\n=== Test accuracy (рядки: мережа/глибина, стовпці: класифікатор)")
    order = list(dict.fromkeys(zip(df.model, df.depth)))  # порядок: як у файлах ознак
    table = df.pivot_table(index=["model", "depth"], columns="clf", values="test_acc").reindex(order)
    table = table[[c for c in CLASSIFIERS if c in table.columns]]
    print(table.round(3).to_string())
    best = df.loc[df.val_acc.idxmax()]
    print(f"\nНайкраща за val: {best.model} / {best.depth} / {best.clf}  "
          f"val={best.val_acc:.3f}  test={best.test_acc:.3f}")
    print(f"Збережено в {out_dir.resolve()}")


if __name__ == "__main__":
    main()
