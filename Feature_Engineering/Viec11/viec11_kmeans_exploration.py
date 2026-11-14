"""Khám phá phân cụm KMeans (Chương 3 — Học không giám sát) trên tập Train.

Chỉ dùng tập Train (Tien_Xu_Li/Viec_7/X_train.csv, y_train.csv); không đọc Test.
KMeans không dùng nhãn khi phân cụm. Nhãn Outcome chỉ được dùng SAU khi phân cụm để mô tả
(tỷ lệ lớp dương trong từng cụm, Adjusted Rand Index), không dùng để chọn k hay tham số.
Tiền xử lý: median imputer, thêm Glucose_BMI_Ratio, StandardScaler (fit trên Train).
"""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("viec11_kmeans_audit.log", encoding='utf-8'), logging.StreamHandler()],
)
logger = logging.getLogger("Viec11_KMeans")
RANDOM_STATE = 42


def main():
    logger.info("BẮT ĐẦU KHÁM PHÁ KMEANS (CHỈ TRAIN, KHÔNG DÙNG NHÃN KHI PHÂN CỤM)")
    path = "../../Tien_Xu_Li/Viec_7/"
    X_train = pd.read_csv(f"{path}X_train.csv")
    y_train = pd.read_csv(f"{path}y_train.csv").squeeze()

    filled = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_train), columns=X_train.columns)
    filled["Glucose_BMI_Ratio"] = filled["Glucose"] / filled["BMI"].clip(lower=1e-6)
    scaled = StandardScaler().fit_transform(filled)

    rows, labels_by_k = [], {}
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(scaled)
        labels_by_k[k] = model.labels_
        rows.append({
            "k": k,
            "Inertia": round(float(model.inertia_), 2),
            "Silhouette": round(float(silhouette_score(scaled, model.labels_)), 4),
            "ARI_voi_Outcome": round(float(adjusted_rand_score(y_train, model.labels_)), 4),
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv("bang_kmeans_do_do.csv", index=False, encoding="utf-8-sig")

    best_k = int(metrics.sort_values("Silhouette", ascending=False).iloc[0]["k"])
    detail = []
    for k in sorted({2, best_k}):
        labels = labels_by_k[k]
        for cluster in range(k):
            mask = labels == cluster
            detail.append({
                "k": k, "Cụm": cluster, "Số_Mẫu": int(mask.sum()),
                "Tỷ_Lệ_Outcome_1": round(float(y_train[mask].mean()), 4),
                "Glucose_TB": round(float(filled.loc[mask, "Glucose"].mean()), 1),
                "BMI_TB": round(float(filled.loc[mask, "BMI"].mean()), 1),
                "Age_TB": round(float(filled.loc[mask, "Age"].mean()), 1),
            })
    pd.DataFrame(detail).to_csv("bang_kmeans_mo_ta_cum.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(metrics["k"], metrics["Inertia"], marker="o")
    axes[0].set_title("Elbow (Inertia)")
    axes[0].set_xlabel("k")
    axes[1].plot(metrics["k"], metrics["Silhouette"], marker="o", color="tab:green")
    axes[1].set_title("Silhouette")
    axes[1].set_xlabel("k")
    fig.suptitle("KMeans trên tập Train (Glucose/BMI, 9 feature đã chuẩn hóa)")
    fig.tight_layout()
    fig.savefig("kmeans_elbow_silhouette.png", dpi=160)
    plt.close(fig)
    logger.info("k có Silhouette cao nhất: %s", best_k)
    logger.info("HOÀN THÀNH KHÁM PHÁ KMEANS.")


if __name__ == "__main__":
    main()
