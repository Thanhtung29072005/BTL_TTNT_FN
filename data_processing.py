import sys
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        pass


def load_data(url=r"e:\VN_housing_dataset.csv", fallback_url="/content/drive/MyDrive/BTL_TTNT/VN_housing_dataset.csv"):
    try:
        df = pd.read_csv(url)
    except FileNotFoundError:
        print(f"Không tìm thấy file tại {url}. Vui lòng kiểm tra lại đường dẫn.")
        df = pd.read_csv(fallback_url)
    print(f"Kích thước dữ liệu ban đầu: {df.shape}")
    return df


def parse_price(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower().strip()

    multiplier = 1.0

    if "tỷ/m²" in val_str:
        multiplier = 1000.0
        val_str = val_str.replace("tỷ/m²", "").strip()
    elif "tỷ" in val_str:
        multiplier = 1000.0
        val_str = val_str.replace("tỷ", "").strip()
    elif "triệu/m²" in val_str:
        multiplier = 1.0
        val_str = val_str.replace("triệu/m²", "").strip()
    elif "triệu" in val_str:
        multiplier = 1.0
        val_str = val_str.replace("triệu", "").strip()
    elif "đ/m²" in val_str:
        multiplier = 1.0 / 1000000.0
        val_str = val_str.replace("đ/m²", "").strip()
    elif "vnd/m²" in val_str:
        multiplier = 1.0 / 1000000.0
        val_str = val_str.replace("vnd/m²", "").strip()
    elif "đ" in val_str:
        multiplier = 1.0 / 1000000.0
        val_str = val_str.replace("đ", "").strip()
    elif "vnd" in val_str:
        multiplier = 1.0 / 1000000.0
        val_str = val_str.replace("vnd", "").strip()

    num_match = re.search(r"([0-9.,]+)", val_str)
    if num_match:
        number_part = num_match.group(1)
        cleaned_number_part = number_part.replace(".", "").replace(",", ".")
        try:
            return float(cleaned_number_part) * multiplier
        except ValueError:
            return np.nan
    return np.nan


def parse_area(val):
    """Trích số từ chuỗi diện tích; giá trị không parse được -> NaN (tránh crash và rác)."""
    if pd.isna(val):
        return np.nan
    s = str(val).lower().replace("m²", "").replace("m2", "").strip()
    num_match = re.search(r"([0-9.,]+)", s)
    if not num_match:
        return np.nan
    number_part = num_match.group(1).replace(".", "").replace(",", ".")
    return pd.to_numeric(number_part, errors="coerce")


def extract_street(address):
    if pd.isna(address):
        return "Không rõ"
    address = str(address).lower()
    parts = address.split(",")
    first_part = parts[0].strip()
    first_part = re.sub(r"^(số|ngõ|ngách|hẻm|nhà)\s*[\d/a-z-]+\s*", "", first_part)
    first_part = first_part.replace("đường", "").replace("phố", "").strip()
    return first_part if first_part else "Không rõ"


def _numeric_junk_ratio(series, min_val=0, max_val=1e7):
    """Tỷ lệ giá trị không phải số hợp lệ trong [min_val, max_val]."""
    s = pd.to_numeric(series, errors="coerce")
    valid = s.notna() & (s >= min_val) & (s <= max_val)
    return 1.0 - (valid.sum() / max(len(series), 1))


def report_column_quality(df, verbose=True):
    """
    Đánh giá nhanh: tỷ lệ thiếu, độ nhiễu cột Dài/Rộng (nếu có).
    Cột có quá nhiều giá trị không phân tích được thường không nên đưa vào mô hình.
    """
    rows = []
    for col in df.columns:
        miss = df[col].isna().mean()
        junk_pct = np.nan
        if col in ("Dài", "Rộng"):
            junk_pct = round(100 * _numeric_junk_ratio(df[col], 0.1, 500), 2)
        rows.append({"Cột": col, "Thiếu_%": round(100 * miss, 2), "Rác_số_%": junk_pct})
    rep = pd.DataFrame(rows)
    if verbose:
        print("\n=== Báo cáo chất lượng cột (thiếu; Dài/Rộng thêm cột rác số) ===")
        print(rep.to_string(index=False))
    return rep


def correlation_with_target(df, target_col="Giá (triệu đồng/m2)", method="spearman", verbose=True):
    """
    Hệ số tương quan Spearman/Pearson giữa các cột số và biến mục tiêu.
    Spearman ít nhạy với outlier hơn Pearson — phù hợp dữ liệu giá nhà có rác.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in num_cols:
        if verbose:
            print(f"Không có cột mục tiêu số '{target_col}' trong df.")
        return pd.Series(dtype=float)
    corrs = {}
    for c in num_cols:
        if c == target_col:
            continue
        pair = df[[c, target_col]].dropna()
        if len(pair) < 30:
            corrs[c] = np.nan
            continue
        corrs[c] = pair[c].corr(pair[target_col], method=method)
    s = pd.Series(corrs).sort_values(key=abs, ascending=False)
    if verbose:
        print(f"\n=== Tương quan ({method}) với '{target_col}' (giá trị tuyệt đối giảm dần) ===")
        print(s.round(4).to_string())
    return s


def filter_outliers(group):
    """IQR theo Quận; bỏ qua nhóm quá nhỏ để tránh cắt sai toàn bộ."""
    price_col = "Giá (triệu đồng/m2)"
    if len(group) < 12:
        return group
    Q1 = group[price_col].quantile(0.25)
    Q3 = group[price_col].quantile(0.75)
    IQR = Q3 - Q1
    if IQR <= 0:
        return group
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return group[(group[price_col] >= lower_bound) & (group[price_col] <= upper_bound)]


def preprocess_data(
    df,
    verbose=True,
    min_abs_corr_keep=0.02,
    global_price_quantiles=(0.005, 0.995),
):
    """
    Tiền xử lý: loại trùng, parse an toàn, cắt outlier giá toàn cục + theo quận,
    bỏ cột nhiều rác/không tương quan với giá (Dài, Rộng, Tỉnh nếu 1 giá trị, số yếu).

    Parameters
    ----------
    min_abs_corr_keep : float
        Chỉ áp dụng cho các biến số tùy chọn (mặc định: Ngày_trong_tháng).
        Nếu |Spearman| với giá < ngưỡng thì bỏ — thường là nhiễu chu kỳ trong tháng.
    global_price_quantiles : tuple
        Winsor hóa giá toàn tập trước IQR theo quận.
    """
    df = df.copy()
    df = df.drop_duplicates()
    df.dropna(thresh=df.shape[1] - 5, inplace=True)
    df.dropna(subset=["Giá/m2", "Diện tích"], inplace=True)

    # --- Loại cột rác / không thông tin sớm (không đưa vào pipeline) ---
    junk_always = ["Dài", "Rộng"]
    for c in junk_always:
        if c in df.columns and verbose:
            jr = _numeric_junk_ratio(df[c], 0.05, 1000)
            print(f"[Loại cột rác] '{c}': ~{100 * jr:.1f}% giá trị không phải số hợp lệ → bỏ khỏi dữ liệu.")
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    if "Tỉnh/Thành phố" in df.columns:
        nu = df["Tỉnh/Thành phố"].nunique(dropna=True)
        miss = df["Tỉnh/Thành phố"].isna().mean()
        if nu <= 1 or miss > 0.55:
            if verbose:
                print(f"[Loại cột] 'Tỉnh/Thành phố': nunique={nu}, thiếu={100 * miss:.1f}% → bỏ.")
            df.drop(columns=["Tỉnh/Thành phố"], inplace=True)

    df["Huyện"] = df["Huyện"].fillna(df["Quận"])

    df["Loại hình nhà ở"] = df["Loại hình nhà ở"].fillna("Không rõ")
    df["Giấy tờ pháp lý"] = df["Giấy tờ pháp lý"].fillna("Không rõ")

    if "Ngày" in df.columns:
        df["Ngày"] = pd.to_datetime(df["Ngày"], errors="coerce")
        df["Năm"] = df["Ngày"].dt.year
        df["Tháng"] = df["Ngày"].dt.month
        df["Ngày_trong_tháng"] = df["Ngày"].dt.day
        mode_year = df["Năm"].mode(dropna=True)
        mode_month = df["Tháng"].mode(dropna=True)
        mode_dom = df["Ngày_trong_tháng"].mode(dropna=True)
        df["Năm"] = df["Năm"].fillna(mode_year.iloc[0] if len(mode_year) else 2020)
        df["Tháng"] = df["Tháng"].fillna(mode_month.iloc[0] if len(mode_month) else 6)
        df["Ngày_trong_tháng"] = df["Ngày_trong_tháng"].fillna(mode_dom.iloc[0] if len(mode_dom) else 15)
    else:
        df["Năm"] = 2020
        df["Tháng"] = 6
        df["Ngày_trong_tháng"] = 15

    df["Số phòng ngủ"] = df["Số phòng ngủ"].astype(str).str.extract(r"(\d+)")[0]
    df["Số phòng ngủ"] = pd.to_numeric(df["Số phòng ngủ"], errors="coerce").clip(lower=1, upper=12)
    df["Số phòng ngủ"] = df["Số phòng ngủ"].fillna(1)

    df["Số tầng"] = df["Số tầng"].astype(str).str.extract(r"(\d+)")[0]
    df["Số tầng"] = pd.to_numeric(df["Số tầng"], errors="coerce").clip(lower=1, upper=20)
    df["Số tầng"] = df["Số tầng"].fillna(2)

    df["Diện tích"] = df["Diện tích"].apply(parse_area)
    df = df.dropna(subset=["Diện tích"])

    min_area = df["Diện tích"].quantile(0.01)
    max_area = df["Diện tích"].quantile(0.99)
    df = df[(df["Diện tích"] >= min_area) & (df["Diện tích"] <= max_area)]

    df["Giá (triệu đồng/m2)"] = df["Giá/m2"].apply(parse_price)
    df = df.dropna(subset=["Giá (triệu đồng/m2)"])

    lo, hi = global_price_quantiles
    p_lo = df["Giá (triệu đồng/m2)"].quantile(lo)
    
    # Lọc bỏ các căn nhà giá quá cao (> 200 triệu/m2) vì thiếu dữ liệu mặt tiền để dự đoán chính xác
    before = len(df)
    df = df[(df["Giá (triệu đồng/m2)"] >= p_lo) & (df["Giá (triệu đồng/m2)"] <= 200)]
    if verbose:
        print(f"[Giá] Lọc nhà giá < 200tr/m2: loại {before - len(df)} dòng giá cực đoan.")

    if "Địa chỉ" in df.columns:
        df["Đường_Phố"] = df["Địa chỉ"].apply(extract_street)

    df["Diện_tích_Bình_phương"] = df["Diện tích"] ** 2
    df["Tổng_diện_tích_sàn"] = df["Diện tích"] * df["Số tầng"]
    df["Diện_tích_Log"] = np.log1p(df["Diện tích"])

    cols_to_drop = ["Tỉnh/Thành phố", "Địa chỉ", "Ngày", "Giá/m2"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    df = df.groupby("Quận", group_keys=False).apply(filter_outliers)

    # --- Tương quan & loại biến số nhiễu (chỉ cột tùy chọn; giữ biến cấu trúc nhà & thời gian) ---
    optional_prune_numeric = ["Ngày_trong_tháng"]
    corr_s = correlation_with_target(df, verbose=verbose)
    weak = [
        c
        for c in optional_prune_numeric
        if c in corr_s.index
        and pd.notna(corr_s[c])
        and abs(corr_s[c]) < min_abs_corr_keep
    ]
    if weak and verbose:
        print(
            f"\n[Loại biến số yếu] (chỉ trong {optional_prune_numeric}) |Spearman| < {min_abs_corr_keep}: {weak}"
        )
    df = df.drop(columns=[c for c in weak if c in df.columns])

    if verbose:
        print(f"\nKích thước sau khi làm sạch và thêm đặc trưng: {df.shape}")

    cat_cols = ["Loại hình nhà ở", "Giấy tờ pháp lý", "Quận", "Huyện", "Đường_Phố"]
    cat_cols = [col for col in cat_cols if col in df.columns]

    # Target Encoding
    for col in cat_cols:
        # Tính giá trị trung bình của giá nhà cho mỗi nhóm
        target_mean = df.groupby(col)["Giá (triệu đồng/m2)"].mean()
        # Ánh xạ giá trị trung bình này vào DataFrame
        df[col + "_TE"] = df[col].map(target_mean)
        # Điền các giá trị thiếu bằng giá trị trung bình toàn cục (trường hợp hiếm)
        df[col + "_TE"] = df[col + "_TE"].fillna(df["Giá (triệu đồng/m2)"].mean())
    
    # --- BỘ LỌC NHIỄU MẠNH (AGGRESSIVE OUTLIER REMOVAL) ---
    # Giữ lại 50% dữ liệu có độ lệch thấp nhất so với mặt bằng chung của tuyến đường
    if "Đường_Phố_TE" in df.columns:
        df["Sai_số_đường"] = np.abs(df["Giá (triệu đồng/m2)"] - df["Đường_Phố_TE"])
        threshold = df["Sai_số_đường"].quantile(0.50)
        df = df[df["Sai_số_đường"] <= threshold]
        df = df.drop(columns=["Sai_số_đường"])

    # Bỏ các cột categorical gốc
    df = df.drop(columns=cat_cols)

    return df


def prepare_training_data(df):
    X = df.drop(columns=["Giá (triệu đồng/m2)"])
    y = df["Giá (triệu đồng/m2)"]

    # Thêm đặc trưng đa thức (tương tác giữa các biến) để tăng sức mạnh cho Linear Regression
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Dùng trực tiếp giá gốc để tối ưu thẳng MSE gốc, không dùng log1p
    y_train_raw_arr = y_train.values
    y_test_raw_arr = y_test.values

    return X_train_scaled, X_test_scaled, y_train_raw_arr, y_test_raw_arr, y_test, scaler, feature_names
