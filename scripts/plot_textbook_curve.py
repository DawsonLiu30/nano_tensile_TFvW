import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

def main():
    # 1. 自動尋找最新的結果資料夾
    result_dirs = glob.glob("results/paper_reproduction_111_Al_*")
    if not result_dirs:
        print("找不到結果資料夾！")
        return
    latest_dir = max(result_dirs, key=os.path.getmtime)
    summary_path = os.path.join(latest_dir, "summary.csv")

    print(f"正在讀取數據：{summary_path}")
    
    # 2. 讀取數據
    df = pd.read_csv(summary_path)
    
    # 移除有 NaN 的行，確保畫圖順利
    df = df.dropna(subset=['strain', 'eng_stress_top_GPa'])

    # 換算應變為百分比 (%)
    strain_pct = df['strain'] * 100
    
    # 取得工程應力 (GPa) 
    # DFTpy 輸出的張力有時會帶負號，我們取絕對值來符合教科書慣例
    stress_gpa = np.abs(df['eng_stress_top_GPa'])

    # 3. 找出降伏點 (最大應力處)
    max_idx = stress_gpa.idxmax()
    yield_strain = strain_pct.iloc[max_idx]
    yield_stress = stress_gpa.iloc[max_idx]

    # 4. 開始畫教科書級別的圖表
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # 畫出主曲線
    ax.plot(strain_pct, stress_gpa, marker='o', markersize=6, linestyle='-', 
            linewidth=2.5, color='#1f77b4', label='Tensile Test Data')

    # 標記降伏點 (Yield Point)
    ax.scatter([yield_strain], [yield_stress], color='red', s=100, zorder=5)
    ax.annotate(f'Yield Strength\n({yield_strain:.1f}%, {yield_stress:.2f} GPa)', 
                xy=(yield_strain, yield_stress), 
                xytext=(yield_strain + 1.5, yield_stress - 0.5),
                arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                fontsize=12, fontweight='bold', color='red')

    # 用顏色區分彈性區與塑性區 (以降伏點為界)
    ax.axvspan(0, yield_strain, color='lightgreen', alpha=0.2, label='Elastic Region')
    ax.axvspan(yield_strain, strain_pct.max() + 1, color='lightcoral', alpha=0.2, label='Plastic Region')

    # 美化圖表
    ax.set_title('Stress-Strain Curve of [111] Al Nanowire (DFT Simulation)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Engineering Strain (%)', fontsize=14)
    ax.set_ylabel('Engineering Stress (GPa)', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=12, loc='lower right')
    
    # 設定 X 軸稍微多一點點空間讓圖更好看
    ax.set_xlim(0, strain_pct.max() + 2)
    ax.set_ylim(0, stress_gpa.max() + 1.5)

    plt.tight_layout()
    
    # 存檔
    output_png = "stress_strain_curve.png"
    plt.savefig(output_png, dpi=300)
    print(f"✅ 圖表已成功繪製並存檔至：{output_png}")

if __name__ == "__main__":
    main()