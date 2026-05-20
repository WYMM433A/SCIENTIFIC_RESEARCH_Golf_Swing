import pandas as pd
import numpy as np
from typing import Dict, List
from .angles import extract_metrics_from_row
from .benchmarks import BENCHMARKS, get_status

class SwingBiomechanicsEvaluator:
    def __init__(self, player_level: str = "amateur"):
        self.player_level = player_level

    def evaluate(self, poses_df: pd.DataFrame, phase_keyframes: Dict[str, int]) -> Dict:
        """
        Phân tích toàn bộ swing dựa trên các keyframes đã detect.
        phase_keyframes: ví dụ {'Top': 124, 'Impact': 150, 'Address': 20}
        """
        report = {
            "summary": {},
            "detailed_metrics": [],
            "priority_fixes": []
        }
        
        # 1. Xác định Baseline Target Line từ Address (nếu có) để tính góc xoay tương đối
        baseline_h_vec = None
        if 'Address' in phase_keyframes:
            addr_idx = phase_keyframes['Address']
            if addr_idx < len(poses_df):
                row = poses_df.iloc[addr_idx]
                # Tạo vector hông trên mặt phẳng X-Z làm chuẩn
                # Đảm bảo các cột tồn tại trước khi truy cập
                if not all(col in row for col in ['left_hip_x', 'left_hip_z', 'right_hip_x', 'right_hip_z']):
                    print(f"⚠️ Cảnh báo: Thiếu dữ liệu hông tại frame Address ({addr_idx}). Không thể xác định baseline.")
                else:
                    h_l = np.array([row['left_hip_x'], row['left_hip_z']])
                    h_r = np.array([row['right_hip_x'], row['right_hip_z']])
                    baseline_h_vec = h_r - h_l
            else:
                print(f"⚠️ Cảnh báo: Frame Address ({addr_idx}) nằm ngoài phạm vi poses_df.")

        # 2. Tính metrics cho từng frame quan trọng
        key_metrics = {}
        for phase, frame_idx in phase_keyframes.items():
            if frame_idx < len(poses_df):
                row = poses_df.iloc[frame_idx].to_dict()
                metrics_for_frame = extract_metrics_from_row(row, baseline_h_vec=baseline_h_vec)
                if metrics_for_frame is not None:
                    key_metrics[phase] = metrics_for_frame
                else:
                    print(f"⚠️ Cảnh báo: Không thể trích xuất metrics cho pha '{phase}' tại frame {frame_idx} do thiếu landmark.")
            else:
                print(f"⚠️ Cảnh báo: Frame '{phase}' ({frame_idx}) nằm ngoài phạm vi poses_df.")

        # 3. Phân tích Temporal (Head Stability - Cải tiến)
        # Thu thập tất cả vị trí đầu (nose) từ các frame có sẵn
        all_head_positions = []
        for _, row_data in poses_df.iterrows():
            # Sử dụng extract_metrics_from_row để lấy head_pos, nhưng chỉ cần phần head_pos
            # Tránh gọi extract_metrics_from_row quá nhiều lần nếu nó nặng
            # Thay vào đó, trích xuất trực tiếp từ row nếu có thể
            if 'nose_x' in row_data and 'nose_y' in row_data and 'nose_z' in row_data and 'nose_v' in row_data and row_data['nose_v'] >= 0.5:
                all_head_positions.append(np.array([row_data['nose_x'], row_data['nose_y'], row_data['nose_z']]))
        
        head_stability_val = 0.0
        if len(all_head_positions) > 1: # Cần ít nhất 2 điểm để tính độ lệch chuẩn
            head_pts = np.array(all_head_positions)
            # Tính độ lệch chuẩn của vị trí đầu theo trục X, Y, Z và lấy norm của vector std
            # Hoặc chỉ lấy std của một trục quan trọng, ví dụ trục X (ngang)
            head_stability_val = np.linalg.norm(np.std(head_pts, axis=0)) * 100 # Nhân 100 để chuyển sang cm giả định
        
        # Thêm global metrics vào key_metrics
        key_metrics["Global"] = {"head_stability": head_stability_val}

        # 4. So sánh với benchmarks & Tính điểm weighted
        scores = []
        total_weight = 0
        for metric_id, cfg in BENCHMARKS.items():
            phase = cfg["phase"]
            
            # Kiểm tra xem pha và metric có tồn tại trong dữ liệu đã trích xuất không
            if phase in key_metrics and metric_id in key_metrics[phase]:
                val = key_metrics[phase][metric_id]
                
                # Bỏ qua nếu giá trị là None (do thiếu landmark)
                if val is None:
                    continue
                
                status, color = get_status(val, cfg, self.player_level)
                weight = cfg.get("priority_weight", 1.0)
                
                metric_result = {
                    "name": cfg["label"],
                    "value": round(val, 1),
                    "unit": cfg["unit"],
                    "status": status,
                    "color": color,
                    "ideal_range": f"{cfg[self.player_level]['min']}-{cfg[self.player_level]['max']}",
                    "feedback": cfg["hint"] if status != "Good" else "Tuyệt vời!"
                }
                report["detailed_metrics"].append(metric_result)
                
                score = 100 if status == "Good" else (60 if status == "Fair" else 20)
                scores.append(score * weight)
                total_weight += weight
                
                # 5. Logic Priority Fixes dựa trên trọng số và mức độ sai lệch
                if status != "Good":
                    deviation = abs(val - cfg[self.player_level]["ideal"])
                    report["priority_fixes"].append({
                        "score_impact": deviation * weight, # Tính toán tác động của lỗi
                        "issue": cfg["label"],
                        "advice": cfg["hint"]
                    })

        # Sắp xếp lỗi theo tác động (score_impact) giảm dần
        report["priority_fixes"].sort(key=lambda x: x["score_impact"], reverse=True)
        
        # Tổng kết điểm tổng thể
        report["summary"] = {
            "overall_score": int(np.sum(scores) / total_weight) if total_weight > 0 else 0,
            "player_level": self.player_level,
            "total_metrics_analyzed": len(scores)
        }
        
        return report