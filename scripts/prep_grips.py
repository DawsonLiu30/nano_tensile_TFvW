import numpy as np
from ase.io import read
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="為奈米線建立頂部與底部的固定端 (Grips)")
    parser.add_argument("--init", default="init_111_nanowire.vasp", help="輸入的 VASP 結構檔")
    parser.add_argument("--thickness", type=float, default=10.0, help="Grip 的厚度 (單位: Å)")
    parser.add_argument("--outdir", default=".", help="npy 檔案的輸出目錄")
    args = parser.parse_args()

    print(f"載入結構: {args.init}")
    atoms = read(args.init)
    
    # 取得所有原子在 Z 軸的座標
    z_positions = atoms.get_positions()[:, 2]
    z_min = z_positions.min()
    z_max = z_positions.max()
    
    print(f"奈米線 Z 軸範圍: {z_min:.2f} Å 到 {z_max:.2f} Å (總長: {z_max - z_min:.2f} Å)")

    # 篩選底部與頂部原子
    bottom_mask = z_positions <= (z_min + args.thickness)
    top_mask = z_positions >= (z_max - args.thickness)

    # 轉換為 index 陣列
    bottom_idx = np.where(bottom_mask)[0]
    top_idx = np.where(top_mask)[0]

    if len(bottom_idx) == 0 or len(top_idx) == 0:
        raise ValueError("錯誤：找不到 Grip 原子，請檢查厚度設定是否合理！")

    # 確保輸出目錄存在
    os.makedirs(args.outdir, exist_ok=True)

    # 儲存為 numpy 陣列格式供 main.py 讀取
    np.save(os.path.join(args.outdir, "bottom_idx.npy"), bottom_idx)
    np.save(os.path.join(args.outdir, "top_idx.npy"), top_idx)

    print(f"成功萃取 Grip！")
    print(f"-> 底部原子數量 (Bottom): {len(bottom_idx)}")
    print(f"-> 頂部原子數量 (Top): {len(top_idx)}")
    print(f"-> 檔案已儲存至: {args.outdir}")

if __name__ == "__main__":
    main()