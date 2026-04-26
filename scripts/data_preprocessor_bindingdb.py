# data_preprocessor_bindingdb.py

import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolStandardize
import logging
import math
from tqdm import tqdm
import os

# --- 配置 ---
# 请确保您下载的BindingDB TSV文件名与此匹配
INPUT_FILE = './data/BindingDB_All.tsv' # 或者您下载的最新版本文件名
# 我们期望的pActivity阈值
MIN_PACTIVITY = 6.0
# 输出文件名
OUTPUT_FILE = 'protein_ligand_pactivity.csv'

# --- 初始化工具 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
normalizer = MolStandardize.normalize.Normalizer()
tqdm.pandas()

def smiles_is_valid_and_standardize(smiles):
    """检查SMILES是否有效，如果有效则进行标准化并返回。否则返回None。"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = normalizer.normalize(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

def process_bindingdb_data(input_tsv, output_csv, min_pactivity):
    """
    高效地处理本地的BindingDB数据文件（适配2024+版本），以提取高质量的蛋白-配体对。
    """
    
    if not os.path.exists(input_tsv):
        logging.error(f"错误：输入文件 '{input_tsv}' 未找到。")
        logging.error("请从 BindingDB 官网下载 'BindingDB_All_2D_... .tsv.zip' 文件，")
        logging.error("解压后将其放在与此脚本相同的目录下。")
        return

    # --- 核心修正：严格使用您提供的、真实的、最新的列名 ---
    COL_SMILES = 'Ligand SMILES'
    '''
    如果蛋白质有多个链（例如 PDB 结构中 ABC 链），BindingDB 只记录了其中一条链的序列，
    默认是 chain A 或实验活性对应的那一条。
    '''
    COL_SEQ = 'BindingDB Target Chain Sequence 1'  # 修正：特指第一条链的序列
    COL_CHAINS = 'Number of Protein Chains in Target (>1 implies a multichain complex)' # 修正：P为大写
    COLS_ACTIVITY = ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']
    
    required_cols = [COL_SMILES, COL_SEQ, COL_CHAINS] + COLS_ACTIVITY
    
    logging.info(f"正在从 '{input_tsv}' 加载数据（适配最新列名）...")
    try:
        # 使用 on_bad_lines='skip' 来跳过任何格式错误的行，增加稳健性
        df = pd.read_csv(input_tsv, sep='\t', usecols=required_cols, on_bad_lines='skip')
    except ValueError as e:
        logging.error(f"读取TSV文件时出错：列名不匹配。请确认 '{input_tsv}' 文件包含以下列: {required_cols}")
        logging.error(f"原始错误: {e}")
        return
    except Exception as e:
        logging.error(f"读取TSV文件时发生未知错误。错误: {e}")
        return

    logging.info(f"原始记录数: {len(df)}")
    
    # a. 移除关键信息缺失的行
    df.dropna(subset=[COL_SMILES, COL_SEQ], inplace=True)
    logging.info(f"移除NA后的记录数: {len(df)}")
    
    # b. 筛选单一蛋白靶点
    df['Chains'] = pd.to_numeric(df[COL_CHAINS], errors='coerce')
    df = df[df['Chains'] == 1.0]
    logging.info(f"筛选单一蛋白靶点后的记录数: {len(df)}")

    # 将所有活性列转换为数字
    for col in COLS_ACTIVITY:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 合并活性值：优先使用Ki > Kd > IC50 > EC50
    df['activity_nM'] = df['Ki (nM)'].fillna(df['Kd (nM)']).fillna(df['IC50 (nM)']).fillna(df['EC50 (nM)'])
    df.dropna(subset=['activity_nM'], inplace=True)
    
    df = df[df['activity_nM'] > 0]
    
    df['p_activity'] = df['activity_nM'].apply(lambda x: 9 - math.log10(x))
    
    # c. 根据pActivity进行最终过滤
    df = df[df['p_activity'] >= min_pactivity]
    logging.info(f"筛选 pActivity >= {min_pactivity} 后的记录数: {len(df)}")

    if df.empty:
        logging.warning("在过滤后，没有剩余的数据。请检查过滤条件。")
        return

    logging.info("正在标准化SMILES...")
    df['ligand_smiles'] = df[COL_SMILES].progress_apply(smiles_is_valid_and_standardize)
    df.dropna(subset=['ligand_smiles'], inplace=True)

    final_df = df[[COL_SEQ, 'ligand_smiles', 'p_activity']].copy()
    final_df.rename(columns={COL_SEQ: 'protein_sequence'}, inplace=True)
    
    final_df.drop_duplicates(subset=['protein_sequence', 'ligand_smiles'], inplace=True)
    final_df['p_activity'] = final_df['p_activity'].round(3)

    if final_df.empty:
        logging.warning("在最终处理后，未能得到任何有效的蛋白-配体对。")
        return

    final_df.to_csv(output_csv, index=False)
    logging.info(f"✅ 数据保存至 {output_csv}，共 {len(final_df)} 条独特的蛋白-配体对。")
    print("\n--- 数据预览 ---")
    print(final_df.head())


if __name__ == '__main__':
    process_bindingdb_data(
        input_tsv=INPUT_FILE,
        output_csv=OUTPUT_FILE,
        min_pactivity=MIN_PACTIVITY
    )