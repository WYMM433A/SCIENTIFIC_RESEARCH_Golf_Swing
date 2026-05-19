import pandas as pd
import sys
import os
from pathlib import Path

# Thêm project root vào path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.biomechanics import SwingBiomechanicsEvaluator

def run_test_on_csv(csv_path, player_level="amateur"):
    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy file: {csv_path}")
        return

    # 1. Load dữ liệu pose
    poses_df = pd.read_csv(csv_path)
    print(f"✓ Đã load {len(poses_df)} frames từ {csv_path}")

    # 2. Giả lập keyframes (Trong thực tế, cái này lấy từ model Neural Network)
    # Bạn có thể thay đổi các số frame này tùy theo video của bạn
    keyframes = {
        'Address': 0,
        'Top': int(len(poses_df) * 0.4),
        'Impact': int(len(poses_df) * 0.6)
    }
    print(f"🔍 Testing với keyframes: {keyframes}")

    # 3. Khởi tạo bộ đánh giá
    evaluator = SwingBiomechanicsEvaluator(player_level=player_level)
    
    # 4. Chạy đánh giá
    report = evaluator.evaluate(poses_df, keyframes)

    # 5. Hiển thị kết quả
    print("\n" + "="*50)
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ (Level: {player_level})")
    print("="*50)
    print(f"Điểm tổng thể: {report['summary']['overall_score']}/100")
    print("-" * 30)
    for metric in report['detailed_metrics']:
        status_icon = "✅" if metric['status'] == "Good" else "⚠️" if metric['status'] == "Fair" else "❌"
        print(f"{status_icon} {metric['name']}: {metric['value']}{metric['unit']} ({metric['status']})")
        print(f"   💡 Feedback: {metric['feedback']}")

if __name__ == "__main__":
    # Thay thế đường dẫn này bằng một file CSV bạn đã trích xuất được
    sample_csv = os.path.join(PROJECT_ROOT, "data", "golfdb_poses", "0_poses.csv")
    run_test_on_csv(sample_csv)