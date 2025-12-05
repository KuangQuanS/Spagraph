import spagraph as spg
import time

for i in [8, 9, 12, 13, 14, 15, 21, 30, 31]:
    dataset_number = str(i)
    
    print(f"\n{'='*60}")
    print(f"正在处理 Dataset {dataset_number} / 32 ...")
    print(f"{'='*60}\n")
    
    # ================= 定义路径 =================
    sc_file = f"/mnt/d/ST_Graduation_Project_data/database/SimualtedSpatalData/dataset{dataset_number}/scRNA.h5ad"
    st_file = f"/mnt/d/ST_Graduation_Project_data/database/SimualtedSpatalData/dataset{dataset_number}/Spatial.h5ad"
    output_dir = f"./deconv_results/DATA{dataset_number}"

    art = spg.vae(sc_file=sc_file, st_file=st_file)
    res = spg.deconv(vae=art, output_dir=output_dir)       
    print(f"\nDataset {dataset_number} 完成: {res['n_clusters']} clusters, Pearson={res['best_pearson']:.4f}")

    time.sleep(15)