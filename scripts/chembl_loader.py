# chembl__loader.py
#读取chembl的SDF库，并做smiles转化

import logging
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class chembl_Loader:
    """
    加载并筛选chembl_先导化合物库中的分子（SMILES）。

    功能包括：
    1. 从文件中读取SMILES字符串。
    2. 去除重复的分子。
    3. 根据分子量（MW）和LogP值进行过滤。
    4. 验证SMILES的有效性。
    """

    def __init__(self,
                 min_mw: float = 200.0,
                 max_mw: float = 600.0,
                 min_logp: float = -2.0,
                 max_logp: float = 6.0):
        """
        初始化加载器并设置筛选参数。

        Args:
            min_mw (float): 最小分子量。
            max_mw (float): 最大分子量。
            min_logp (float): 最小LogP值。
            max_logp (float): 最大LogP值。
        """
        self.min_mw = min_mw
        self.max_mw = max_mw
        self.min_logp = min_logp
        self.max_logp = max_logp
        logging.info(f"chembl_Loader initialized with MW range [{min_mw}, {max_mw}] and LogP range [{min_logp}, {max_logp}]")

    def load_from_file(self, file_path: str, limit: Optional[int] = None) -> List[str]:
        """
        从SMILES文件加载并筛选分子。

        Args:
            file_path (str): 包含SMILES字符串的文件路径（每行一个SMILES）。
            limit (Optional[int]): 要处理的最大分子数量，用于快速测试。

        Returns:
            List[str]: 经过筛选和去重后的有效SMILES列表。
        """
        logging.info(f"Loading molecules from {file_path}...")
        
        with open(file_path, 'r') as f:
            smiles_list = [line.strip().split()[0] for line in f]

        if limit:
            smiles_list = smiles_list[:limit]

        logging.info(f"Loaded {len(smiles_list)} raw SMILES strings.")
        
        unique_smiles = list(set(smiles_list))
        logging.info(f"Found {len(unique_smiles)} unique SMILES strings.")

        filtered_smiles = []
        for i, smi in enumerate(unique_smiles):
            if i > 0 and i % 1000 == 0:
                logging.info(f"Processing SMILES {i}/{len(unique_smiles)}...")
            
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logging.warning(f"Could not parse SMILES: {smi}. Skipping.")
                continue

            # 计算描述符并进行筛选
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)

            if not (self.min_mw <= mw <= self.max_mw):
                continue
            if not (self.min_logp <= logp <= self.max_logp):
                continue
                
            filtered_smiles.append(smi)

        logging.info(f"Finished filtering. Returning {len(filtered_smiles)} molecules.")
        return filtered_smiles

# --- 示例用法 ---
if __name__ == '__main__':
    import os; 
    print('当前Python工作目录:', os.getcwd())

    dummy_file = "./data/chembl_leads_10k.smi"

    # 初始化并使用加载器
    loader = chembl_Loader(min_mw=200, max_mw=500, min_logp=-1, max_logp=5)
    valid_molecules = loader.load_from_file(dummy_file)

    print("\n--- Chembl Loader Results ---")
    print(f"Filtered molecules ({len(valid_molecules)}):")
    for smi in valid_molecules:
        mol = Chem.MolFromSmiles(smi)
        print(f"  - SMILES: {smi}, MW: {Descriptors.MolWt(mol):.2f}, LogP: {Descriptors.MolLogP(mol):.2f}")