import subprocess
import sys
import os

def run_command(cmd):
    print(f"\n🚀 執行指令: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ 錯誤：指令執行失敗，腳本中斷！")
        sys.exit(1)

def main():
    # 實驗矩陣 (可根據需求增減)
    orientations = ["111", "100", "110"]
    diameters_to_test = [1.0, 2.0]
    
    # 空缺設定
    enable_vacancy = True
    vac_conc_pct = 0.5
    
    for o in orientations:
        for d in diameters_to_test:
            print(f"\n{'='*65}")
            print(f"🌟 開始自動化流程：[{o}] 晶向 | 直徑 {d} nm")
            print(f"{'='*65}")
            
            base_vasp = f"init_{o}_Al_{d}nm.vasp"
            
            # --- 步驟 1: 生成完美單晶 ---
            cmd_build = f"python app/ase_nanocrystal.py --diameter {d} --orientation {o}"
            run_command(cmd_build)
            
            # --- 步驟 2: 萃取初始 Grip (厚度 4.0 Å) ---
            cmd_prep = f"python scripts/prep_grips.py --init {base_vasp} --outdir . --thickness 4.0"
            run_command(cmd_prep)
            
            final_vasp = base_vasp
            final_bottom = "bottom_idx.npy"
            final_top = "top_idx.npy"
            case_name = f"Al_{o}_{d}nm_perfect"

            # --- 步驟 3: 製造空缺 (修正檔案覆蓋問題) ---
            if enable_vacancy:
                # 這裡加入了 o 和 d，確保每個迴圈產出的 Tag 都是唯一的！
                vac_tag = f"vac_{o}_{d}nm_{vac_conc_pct}pct"
                case_name = f"Al_{o}_{d}nm_vac{vac_conc_pct}"
                
                cmd_vac = (
                    f"PYTHONPATH=. python app/make_vacancy.py "
                    f"--input {base_vasp} "
                    f"--bottom bottom_idx.npy "
                    f"--top top_idx.npy "
                    f"--mode conc "
                    f"--conc-pct {vac_conc_pct} "
                    f"--tag {vac_tag}"
                )
                run_command(cmd_vac)
                
                final_vasp = f"{vac_tag}.vasp"
                final_bottom = f"bottom_idx_{vac_tag}.npy"
                final_top = f"top_idx_{vac_tag}.npy"

            # --- 步驟 4: 啟動拉伸 ---
            cmd_run = (
                f"PYTHONPATH=. python app/main.py "
                f"--case {case_name} "
                f"--workdir . "
                f"--init {final_vasp} "
                f"--pp al.gga.psp "
                f"--bottom-idx {final_bottom} "
                f"--top-idx {final_top} "
                f"--step 0.005 "
                f"--cycles 40 "
                f"--fmax 0.01 "
                f"--relax-steps 150 "
                f"--plot-summary"
            )
            run_command(cmd_run)
            
            print(f"🎉 [{o}] 直徑 {d} nm 模擬完成！")

    print("\n✅ 所有實驗矩陣處理完畢！")

if __name__ == "__main__":
    main()