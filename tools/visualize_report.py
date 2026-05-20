import json
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def create_radar_chart(json_path):
    """
    Parses evaluation.json and generates a normalized Radar Chart for scientific reporting.
    """
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metrics = data['detailed_metrics']
    labels = [m['name'] for m in metrics]
    
    # Normalize data: Set the center of the 'Ideal' range as 1.0
    user_scores = []
    ideal_ring = [1.0] * len(labels) # Ideal baseline ring
    
    for m in metrics:
        val = m['value']
        # Extract numeric mid-point of ideal range
        r = m['ideal_range'].split('-')
        ideal_mid = (float(r[0]) + float(r[1])) / 2
        
        # Calculate ratio (the closer to 1.0, the better)
        # Note: For Head Stability, lower values are better, so we invert the ratio
        if "Head Stability" in m['name']:
            score = ideal_mid / (val + 1e-6)
        else:
            score = val / ideal_mid
        user_scores.append(score)

    # Configure radar chart
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Close the polygon loop
    user_scores += user_scores[:1]
    ideal_ring += ideal_ring[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Plot Ideal Range (Green)
    ax.plot(angles, ideal_ring, color='#2ecc71', linewidth=2, linestyle='--', label='Pro/Amateur Ideal')
    ax.fill(angles, ideal_ring, color='#2ecc71', alpha=0.1)
    
    # Plot User Performance (Red)
    ax.plot(angles, user_scores, color='#e74c3c', linewidth=3, marker='o', label='Your Performance')
    ax.fill(angles, user_scores, color='#e74c3c', alpha=0.3)

    # Configure axes
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    
    # Remove default radial labels and add custom annotations
    ax.set_yticklabels([])
    ax.grid(True, linestyle=':', alpha=0.7)

    # Add title and legend
    plt.title(f"GOLF BIOMECHANICS ANALYSIS\nOverall Score: {data['summary']['overall_score']}/100", 
              size=16, color='#2c3e50', y=1.1, fontweight='bold')
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    # Priority Fix Recommendation
    if data['priority_fixes']:
        top_fix = data['priority_fixes'][0]
        plt.figtext(0.5, 0.02, f"Recommendation: {top_fix['advice']}", 
                    ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

    # Save image
    output_path = json_path.replace('.json', '_radar.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"SUCCESS: Radar chart generated at: {output_path}")
    plt.show()

if __name__ == "__main__":
    # Mặc định chạy thử với kết quả của video
    json_report = os.path.join("data", "metrics", "70_evaluation.json")
    create_radar_chart(json_report)