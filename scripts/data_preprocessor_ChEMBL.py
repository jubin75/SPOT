# data_preprocessor.py
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

def preprocess_protein_ligand_data(input_csv: str, output_csv: str):
    """
    预处理来自ChEMBL的蛋白质-配体数据。
    - 移除无效行
    - 规范化SMILES
    - 过滤掉过长或无效的蛋白质序列
    """
    logging.info(f"Starting preprocessing of {input_csv}...")
    df = pd.read_csv(input_csv)

    # 移除缺少关键信息的行
    df.dropna(subset=['protein_sequence', 'ligand_smiles'], inplace=True)
    logging.info(f"Rows after dropping NA: {len(df)}")

    # 使用tqdm来显示处理进度
    tqdm.pandas(desc="Canonicalizing SMILES")
    
    # 规范化SMILES并移除无效分子
    df['canonical_smiles'] = df['ligand_smiles'].progress_apply(
        lambda smi: Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True) if Chem.MolFromSmiles(smi) else None
    )
    df.dropna(subset=['canonical_smiles'], inplace=True)
    logging.info(f"Rows after canonicalizing SMILES: {len(df)}")

    # 过滤掉异常的蛋白质序列（例如，过短或过长）
    df = df[df['protein_sequence'].str.len().between(50, 2000)]
    logging.info(f"Rows after filtering protein sequences: {len(df)}")

    # 选择最终需要的列并去重
    final_df = df[['protein_sequence', 'canonical_smiles']].drop_duplicates()

    final_df.to_csv(output_csv, index=False)
    logging.info(f"Preprocessing complete. Cleaned data saved to {output_csv} with {len(final_df)} unique pairs.")

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建一个虚拟的原始数据文件用于演示
    dummy_raw_data = {
        'protein_sequence': ["MGAGSAD...", "MKTAYIA..."],
        'ligand_smiles': ["CC(=O)Nc1ccccc1", "c1ccccc1"],
        'activity_value': [8.5, 7.2]
    }
    pd.DataFrame(dummy_raw_data).to_csv("protein_ligand_raw.csv", index=False)
    
    preprocess_protein_ligand_data("protein_ligand_raw.csv", "protein_ligand_clean.csv")