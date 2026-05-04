import numpy as np
import warnings
import joblib
from sklearn.linear_model import LinearRegression

# Import từ các module vừa tạo
from data_processing import load_data, preprocess_data, prepare_training_data, report_column_quality

from evaluation import evaluate, evaluate_classification, visualize_results, plot_loss_curve, plot_scatter
from model import CustomLinearRegression, CustomLogisticRegression
warnings.filterwarnings('ignore')

def main():
    # 1. Tải và tiền xử lý dữ liệu
    print("Đang tải dữ liệu...")
    df_raw = load_data()

    print("Đang phân tích chất lượng cột (thiếu / rác số trên Dài, Rộng)...")
    report_column_quality(df_raw)

    print("Đang tiền xử lý dữ liệu...")
    df_processed = preprocess_data(df_raw)
    
    print("Đang chuẩn bị tập train/test...")
    X_train_scaled, X_test_scaled, y_train_raw, y_test_raw_arr, y_test_raw, scaler, feature_columns = prepare_training_data(df_processed)
    
    # 2. Huấn luyện mô hình tự code
    print("\n--- Bắt đầu huấn luyện mô hình tự code ---")
    custom_model = CustomLinearRegression(lr=0.001, epochs=6000, lambda_=0.01)
    custom_model.fit(X_train_scaled, y_train_raw)
    
    y_pred_custom = custom_model.predict(X_test_scaled)
    
    # ---------------------------------------------------------
    # CHUYỂN ĐỔI SANG BÀI TOÁN PHÂN LOẠI (LOGISTIC REGRESSION)
    # Dự đoán xem căn nhà có thuộc nhóm "Giá Cao" hay không? 
    # (Giá cao = giá > mức trung vị của tập train)
    # ---------------------------------------------------------
    median_price = np.median(y_train_raw)
    y_train_cls = (y_train_raw > median_price).astype(int)
    y_test_cls = (y_test_raw_arr > median_price).astype(int)
    
    print(f"\n[Logistic Regression] Ngưỡng giá phân chia (Median): {median_price:.2f} triệu/m2")
    
    # 4. Huấn luyện mô hình Custom Logistic Regression
    print("--- Bắt đầu huấn luyện Custom Logistic Regression ---")
    logistic_custom = CustomLogisticRegression(lr=0.05, epochs=3000, lambda_=0.01)
    logistic_custom.fit(X_train_scaled, y_train_cls)
    
    y_pred_logist_custom = logistic_custom.predict(X_test_scaled)
    
   
    
    # 5. Đánh giá (Linear Regression)
    evaluate(y_test_raw.values, y_pred_custom, "Custom Linear Regression")
    
    
    # 5b. Đánh giá (Logistic Regression)
    evaluate_classification(y_test_cls, y_pred_logist_custom, "Custom Logistic Regression")
    
    
    # 6. Export mô hình
    export_data = {
        "custom_linear_model": {
            "w": custom_model.w,
            "b": custom_model.b
        },
        "custom_logistic_model": {
            "w": logistic_custom.w,
            "b": logistic_custom.b
        },  
        "scaler": scaler,
        "feature_columns": list(feature_columns)
    }
    joblib.dump(export_data, "linear_regression_backend.pkl")
    print("\n[OK] Đã xuất toàn bộ dữ liệu mô hình ra file 'linear_regression_backend.pkl'")

    # 5. Trực quan hóa
    print("\nĐang hiển thị biểu đồ... (Hãy tắt cửa sổ biểu đồ để kết thúc chương trình)")
    plot_loss_curve(custom_model.loss_history, "Loss Curve - Custom Linear Regression")
    plot_loss_curve(custom_model.r2_history, "R2 Score Curve - Custom Linear Regression")
    
    plot_loss_curve(logistic_custom.loss_history, "Loss Curve - Custom Logistic Regression")
    plot_loss_curve(logistic_custom.accuracy_history, "Accuracy Curve - Custom Logistic Regression")
    plot_scatter(y_test_raw.values, y_pred_custom, "Scatter Plot: Thực tế vs Dự đoán (Custom Linear Regression)")
    visualize_results(y_test_raw.values, y_pred_custom, "Gia thuc te vs Gia du doan (Custom Linear Regression)")

if __name__ == "__main__":
    main()
