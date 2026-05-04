import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(y_true, y_pred, model_name="Model"):
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n[{model_name}] Kết quả đánh giá trên tập Test:")
    print(f" - R² Score : {r2:.4f} (Càng gần 1 càng tốt)")
    print(f" - RMSE     : {rmse:.4f} (Lỗi trung bình tính theo triệu đồng/m2)")
    print(f" - MAE      : {mae:.4f} (Sai số tuyệt đối trung bình)")
    return mse, rmse, mae, r2

def evaluate_classification(y_true, y_pred, model_name="Classification Model"):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n[{model_name}] Kết quả đánh giá phân loại:")
    print(f" - Accuracy  : {acc:.4f} (Độ chính xác tổng thể)")
    print(f" - Precision : {prec:.4f} (Tỉ lệ đoán đúng nhà giá cao)")
    print(f" - Recall    : {rec:.4f} (Khả năng bắt được nhà giá cao)")
    print(f" - F1 Score  : {f1:.4f} (Trung bình điều hòa)")
    return acc, prec, rec, f1

def visualize_results(y_true, y_pred, title="Phân phối Giá thực tế vs Giá dự đoán"):
    plt.figure(figsize=(10, 6))
    
    # Ép kiểu về numpy array 1D để tránh lỗi với sns.kdeplot
    y_true_arr = np.asarray(y_true).flatten()
    y_pred_arr = np.asarray(y_pred).flatten()
    
    sns.kdeplot(x=y_true_arr, label='Giá trị thực tế', fill=True, color='blue', alpha=0.3)
    sns.kdeplot(x=y_pred_arr, label='Dự đoán', fill=True, color='orange', alpha=0.3)
    
    plt.xlim(0, np.percentile(y_true_arr, 95)) 
    plt.xlabel("Giá (triệu đồng/m2)")
    plt.ylabel("Mật độ (Density)")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_loss_curve(history, title="Loss Curve"):
    plt.figure(figsize=(8, 5))
    metric_name = "Loss" if "Loss" in title else ("R2 Score" if "R2" in title else "Accuracy")
    plt.plot(history, label=f'Training {metric_name}', color='red' if "Loss" in title else 'blue')
    plt.xlabel("Epochs (x500)" if metric_name != "Loss" else "Epochs")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_scatter(y_true, y_pred, title="Scatter Plot: Thực tế vs Dự đoán"):
    plt.figure(figsize=(8, 5))
    y_true_arr = np.asarray(y_true).flatten()
    y_pred_arr = np.asarray(y_pred).flatten()
    
    plt.scatter(y_true_arr, y_pred_arr, alpha=0.5, color='blue')
    
    max_val = max(np.max(y_true_arr), np.max(y_pred_arr))
    min_val = min(np.min(y_true_arr), np.min(y_pred_arr))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Đường y=x')
    
    plt.xlabel("Giá thực tế (triệu đồng/m2)")
    plt.ylabel("Giá dự đoán (triệu đồng/m2)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
