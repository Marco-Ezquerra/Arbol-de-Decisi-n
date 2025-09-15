# demo_nodo_arbol.py
# ------------------------------------------------------------
# Demostración de un Árbol de Decisión "hecho a mano" (Nodo)
# + Comparativa con sklearn DecisionTreeClassifier
# + Repetición en bucle (n_iter) con semillas aleatorias distintas
# ------------------------------------------------------------

import argparse
import numpy as np
import pandas as pd
from arbol_modulo import Nodo

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, brier_score_loss
)
from sklearn.tree import DecisionTreeClassifier


# ------------------------------------------------------------
# 1) CARGA DE DATOS
# ------------------------------------------------------------
def cargar_dataset(nombre: str) -> tuple[pd.DataFrame, str, list[str]]:
    nombre = nombre.lower()
    if nombre == "breast_cancer":
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series([data.target_names[i] for i in data.target], name="diagnosis")
        df = pd.concat([X, y], axis=1)
        target_col = "diagnosis"
        feature_cols = list(X.columns)
    elif nombre == "iris":
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series([data.target_names[i] for i in data.target], name="species")
        df = pd.concat([X, y], axis=1)
        target_col = "species"
        feature_cols = list(X.columns)
    else:
        raise ValueError("Dataset no soportado. Usa 'breast_cancer' o 'iris'.")
    return df, target_col, feature_cols


# ------------------------------------------------------------
# 2) UNA EJECUCIÓN (con una seed)
# ------------------------------------------------------------
def run_once(df: pd.DataFrame,
             target_col: str,
             feature_cols: list[str],
             depth: int,
             min_leaf: int,
             max_thr: int | None,
             seed: int,
             dataset_es_binario: bool) -> dict:
    """Entrena y evalúa Nodo y sklearn una vez con la seed dada. Devuelve métricas."""
    # Split estratificado por reproducibilidad
    stratify = df[target_col] if df[target_col].nunique() > 1 else None
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=stratify
    )
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test  = test_df[feature_cols]
    y_test  = test_df[target_col]

    # -------- Nodo (manual) --------
    raiz = Nodo(
        datos=train_df,
        target_col=target_col,
        feature_cols=feature_cols,
        max_thresholds_per_feature=max_thr,
        min_samples_leaf=min_leaf
    )
    raiz.construir_subarbol(profundidad_max=depth)

    y_pred_n = raiz.predecir(X_test)
    Pn = raiz.predict_proba(X_test)
    classes_n = list(raiz.classes_)
    y_test_cat_n = pd.Categorical(y_test, categories=classes_n, ordered=True)
    y_true_int_n = y_test_cat_n.codes

    acc_n = accuracy_score(y_test, y_pred_n)
    ll_n  = log_loss(y_true_int_n, Pn, labels=list(range(len(classes_n))))

    if len(classes_n) == 2:
        # binario
        p_pos_n = Pn[:, 1]  # segunda clase como positiva (mismo orden que classes_)
        y_bin_n = (y_test == classes_n[1]).astype(int)
        auc_n = roc_auc_score(y_bin_n, p_pos_n)
        # Brier sólo para info (no lo devolvemos si no lo necesitas)
        brier_n = brier_score_loss(y_bin_n, p_pos_n)
    else:
        # multiclase
        auc_n = roc_auc_score(y_true_int_n, Pn, multi_class="ovr", average="macro")
        brier_n = np.nan

    # -------- sklearn --------
    sk_clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        min_samples_leaf=min_leaf,
        random_state=seed
    ).fit(X_train, y_train)

    y_pred_s = sk_clf.predict(X_test)
    Ps = sk_clf.predict_proba(X_test)
    classes_s = list(sk_clf.classes_)
    y_test_cat_s = pd.Categorical(y_test, categories=classes_s, ordered=True)
    y_true_int_s = y_test_cat_s.codes

    acc_s = accuracy_score(y_test, y_pred_s)
    ll_s  = log_loss(y_true_int_s, Ps, labels=list(range(len(classes_s))))

    if len(classes_s) == 2:
        p_pos_s = Ps[:, 1]
        y_bin_s = (y_test == classes_s[1]).astype(int)
        auc_s = roc_auc_score(y_bin_s, p_pos_s)
        brier_s = brier_score_loss(y_bin_s, p_pos_s)
    else:
        auc_s = roc_auc_score(y_true_int_s, Ps, multi_class="ovr", average="macro")
        brier_s = np.nan

    return {
        "acc_n": acc_n, "auc_n": auc_n, "ll_n": ll_n, "brier_n": brier_n,
        "acc_s": acc_s, "auc_s": auc_s, "ll_s": ll_s, "brier_s": brier_s,
        "seed": seed
    }


# ------------------------------------------------------------
# 3) MAIN: Bucle de N iteraciones con seeds aleatorias
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Nodo vs sklearn con repetición multi-seed.")
    parser.add_argument("--dataset", type=str, default="breast_cancer",
                        choices=["breast_cancer", "iris"],
                        help="Dataset de ejemplo a usar.")
    parser.add_argument("--depth", type=int, default=5, help="Profundidad máxima del árbol.")
    parser.add_argument("--min_leaf", type=int, default=8, help="Mínimo de muestras por hoja.")
    parser.add_argument("--max_thr", type=int, default=32,
                        help="Máx. thresholds por feature (cuantiles). None = todos.")
    parser.add_argument("--n_iter", type=int, default=50, help="Nº de repeticiones (seeds aleatorias distintas).")
    parser.add_argument("--seed_base", type=int, default=123,
                        help="Semilla base para generar distintas seeds (permite reproducibilidad del experimento).")
    args = parser.parse_args()

    # Datos
    df, target_col, feature_cols = cargar_dataset(args.dataset)
    print("==============================================")
    print("ÁRBOL DE DECISIÓN (Nodo) – DEMO + SKLEARN (multi-seed)")
    print("==============================================")
    print(f"Dataset: {args.dataset}")
    print(f"Filas totales: {len(df)}")
    print(f"Target: {target_col}")
    print(f"Features ({len(feature_cols)}): {feature_cols[:8]}{'...' if len(feature_cols)>8 else ''}")
    print(f"\nParámetros → depth={args.depth} | min_leaf={args.min_leaf} | max_thr={args.max_thr} | n_iter={args.n_iter}")

    # Generamos seeds aleatorias distintas pero reproducibles a partir de seed_base
    rng = np.random.default_rng(args.seed_base)
    seed_list = rng.integers(0, 10**9, size=args.n_iter).tolist()

    # Ejecutamos N veces
    results = []
    for i, seed in enumerate(seed_list, start=1):
        r = run_once(
            df=df,
            target_col=target_col,
            feature_cols=feature_cols,
            depth=args.depth,
            min_leaf=args.min_leaf,
            max_thr=args.max_thr if args.max_thr is not None else None,
            seed=int(seed),
            dataset_es_binario=(df[target_col].nunique() == 2)
        )
        results.append(r)
        print(f"Iteración {i:02d}/{args.n_iter} completada (seed={seed}).")

    res = pd.DataFrame(results)

    # Resumen estadístico
    def fmt(col): 
        return f"{res[col].mean():.4f} ± {res[col].std():.4f}"

    print("\n================  RESUMEN MULTI-SEED (50 iter)  ================")
    print(f"Seeds (primeras 10): {seed_list[:10]}{' ...' if len(seed_list)>10 else ''}")

    print(f"Accuracy → Nodo: {fmt('acc_n')} | Sklearn: {fmt('acc_s')}")
    print(f"AUC      → Nodo: {fmt('auc_n')} | Sklearn: {fmt('auc_s')}")
    print(f"LogLoss  → Nodo: {fmt('ll_n')}  | Sklearn: {fmt('ll_s')}")
    # Brier sólo sentido en binario; si multiclase será NaN
    if not res['brier_n'].isna().all() and not res['brier_s'].isna().all():
        print(f"Brier    → Nodo: {fmt('brier_n')} | Sklearn: {fmt('brier_s')}")

    # % de runs en los que gana Nodo
    pct_acc = (res["acc_n"] > res["acc_s"]).mean()*100
    pct_auc = (res["auc_n"] > res["auc_s"]).mean()*100
    pct_ll  = (res["ll_n"]  < res["ll_s"]).mean()*100  # menor es mejor
    print("\n% de runs donde gana Nodo:")
    print(f"  Accuracy: {pct_acc:.1f}% | AUC: {pct_auc:.1f}% | LogLoss: {pct_ll:.1f}%")

    print("\nListo. ✅")


if __name__ == "__main__":
    main()
