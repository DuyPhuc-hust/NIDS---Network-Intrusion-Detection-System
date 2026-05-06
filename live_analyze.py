import pandas as pd
import joblib
import subprocess
import os
import numpy as np
import sys

# 1. Cấu hình đường dẫn để nhận diện module từ thư mục src
# Điều này đảm bảo script tìm thấy src/data/preprocess.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    from src.data.preprocess import clean_data
    print("[+] Đã kết nối thành công với module tiền xử lý dữ liệu.")
except ImportError:
    print("[!] Lỗi: Không tìm thấy thư mục src hoặc file preprocess.py.")
    print("Hãy đảm bảo bạn đang đặt file này ở thư mục gốc của project.")
    sys.exit(1)

def live_analyze(pcap_path):
    # Đường dẫn tới các file model đã push lên git
    # Dựa trên cấu trúc folder: models/ nằm ở thư mục gốc
    model_path = 'models/xgb.pkl'
    scaler_path = 'models/scaler.pkl'
    le_path = 'models/label_encoder.pkl'

    # Kiểm tra sự tồn tại của model trước khi chạy
    if not all(os.path.exists(p) for p in [model_path, scaler_path, le_path]):
        print(f"[!] Thiếu file tại thư mục models/. Vui lòng kiểm tra lại Git.")
        return

    print(f"\n[+] Đang chuẩn bị phân tích file: {pcap_path}")
    
    # 2. Load model và các công cụ bổ trợ
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    
    # 3. Trích xuất đặc trưng bằng CICFlowMeter (Python version)
    output_csv = "temp_extracted_flow.csv"
    print("[+] Đang trích xuất đặc trưng mạng (PCAP -> CSV)...")
    
    try:
        # Chạy lệnh cicflowmeter từ Terminal
        subprocess.run(["cicflowmeter", "-f", pcap_path, "-c", output_csv], check=True)
    except FileNotFoundError:
        print("[!] Lỗi: Chưa cài đặt cicflowmeter. Hãy chạy 'pip install cicflowmeter'.")
        return
    except Exception as e:
        print(f"[!] Lỗi phát sinh khi trích xuất: {e}")
        return

    # 4. Đọc dữ liệu và xử lý định dạng
    if not os.path.exists(output_csv):
        print("[!] Không có luồng dữ liệu nào được trích xuất.")
        return

    df = pd.read_csv(output_csv)
    # Xóa khoảng trắng thừa trong tên cột (lỗi hay gặp của CICFlowMeter)
    df.columns = df.columns.str.strip()

    # 5. Tiền xử lý (Sử dụng hàm clean_data trong src/data/preprocess.py)
    # Hàm này sẽ xử lý Inf, NaN và chọn đúng các feature cần thiết
    df_cleaned = clean_data(df)
    
    if df_cleaned.empty:
        print("[!] Dữ liệu sau khi làm sạch trống rỗng. Không có gì để dự đoán.")
        return

    # Loại bỏ cột Label nếu có để chuẩn bị đưa vào model
    X = df_cleaned.drop(columns=['Label'], errors='ignore')
    
    # 6. Chuẩn hóa Z-score và Dự đoán
    # Sử dụng scaler cũ từ lúc Train để đảm bảo tính nhất quán
    try:
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        labels = le.inverse_transform(preds)
    except Exception as e:
        print(f"[!] Lỗi khi dự đoán: {e}")
        print("Gợi ý: Kiểm tra số lượng cột dữ liệu trích xuất có khớp với lúc train không.")
        return

    # 7. Hiển thị kết quả ra màn hình
    df_cleaned['Prediction'] = labels
    
    print("\n" + "="*50)
    print(" THỐNG KÊ KẾT QUẢ PHÂN TÍCH NIDS")
    print("="*50)
    print(df_cleaned['Prediction'].value_counts())
    print("-" * 50)

    # Lọc ra các dòng bị nghi ngờ là tấn công
    attacks = df_cleaned[df_cleaned['Prediction'] != 'Normal']
    if not attacks.empty:
        print(f"\n[ALERT] PHÁT HIỆN {len(attacks)} LUỒNG TRAFFIC BẤT THƯỜNG!")
        # Hiển thị thông tin cơ bản để điều tra
        # Lưu ý: Tên cột IP/Port có thể thay đổi tùy bản cicflowmeter, hãy kiểm tra lại
        display_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'Prediction']
        existing_cols = [c for c in display_cols if c in attacks.columns]
        print(attacks[existing_cols].head(20))
    else:
        print("\n[OK] Không phát hiện hành vi xâm nhập nào.")

    # Dọn dẹp file tạm
    if os.path.exists(output_csv):
        os.remove(output_csv)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python3 live_analyze.py <đường_dẫn_file_pcap>")
    else:
        live_analyze(sys.argv[1])