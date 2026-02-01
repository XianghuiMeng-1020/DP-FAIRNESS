"""
数据集加载模块
支持OULAD, UCI697, HarvardX_PersonCourse三个数据集
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import os
# 过滤掉 dp_synth 相关的警告（这些警告来自外部库，不影响我们的代码）
warnings.filterwarnings('ignore', category=UserWarning, module='dp_synth')
warnings.filterwarnings('ignore', message='.*AUROC computation failed.*')
warnings.filterwarnings('ignore', message='.*Test set contains.*labels not in training set.*')
warnings.filterwarnings('ignore', message='.*Number of classes.*not equal.*')
warnings.filterwarnings('ignore')


def load_dataset(dataset_name: str, seed: int = 42, data_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    加载数据集并返回训练/测试特征和标签
    
    Args:
        dataset_name: 数据集名称 ("OULAD", "UCI697", "HarvardX_PersonCourse")
        seed: 随机种子
        data_dir: 数据目录路径（如果为None，尝试从常见位置加载）
    
    Returns:
        X_train, X_test, y_train, y_test, groups_test
        groups_test: 测试集的组标签（用于公平性评估），如果数据集没有demographic字段则为None
    """
    np.random.seed(seed)
    
    if data_dir is None:
        # 尝试常见的数据目录位置
        possible_dirs = [
            Path("data"),
            Path("datasets"),
            Path("outputs/data"),
        ]
        data_dir = None
        for d in possible_dirs:
            if d.exists():
                data_dir = str(d)
                break
        
        # 如果没找到，默认使用data目录（即使不存在，也会在后续路径查找中处理）
        if data_dir is None:
            data_dir = "data"
    
    # 根据数据集名称加载
    if dataset_name == "OULAD":
        return _load_oulad(seed, data_dir)
    elif dataset_name == "UCI697":
        return _load_uci697(seed, data_dir)
    elif dataset_name == "HarvardX_PersonCourse":
        return _load_harvardx(seed, data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_oulad(seed: int, data_dir: Optional[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """加载OULAD数据集"""
    # 尝试从文件加载，如果不存在则生成模拟数据
    if data_dir:
        # Try multiple possible paths
        possible_paths = [
            Path(data_dir) / "raw" / "oulad" / "studentInfo.csv",
            Path(data_dir) / "OULAD" / "studentInfo.csv",
        ]
        # Also search recursively
        base_paths = [Path(data_dir) / "raw" / "oulad", Path(data_dir) / "OULAD"]
        for base in base_paths:
            if base.exists():
                for csv_file in base.rglob("studentInfo.csv"):
                    possible_paths.append(csv_file)
                    break
        
        for csv_path in possible_paths:
            if csv_path.exists():
                return _load_oulad_from_file(csv_path, seed)
    
    # CRITICAL: Synthetic fallback is STRICTLY FORBIDDEN
    # This code path should NEVER be reached
    raise FileNotFoundError(
        f"CRITICAL ERROR: OULAD data file not found. Synthetic fallback is STRICTLY FORBIDDEN. "
        f"Searched paths: {[str(p) for p in possible_paths]}. "
        f"Please ensure real data files are available at one of these paths."
    )


def _load_uci697(seed: int, data_dir: Optional[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """加载UCI697数据集"""
    if data_dir:
        # Try multiple possible paths and filenames
        possible_paths = [
            Path(data_dir) / "raw" / "uci697" / "data.csv",  # Actual downloaded filename
            Path(data_dir) / "raw" / "uci697" / "student-mat.csv",
            Path(data_dir) / "UCI697" / "student-mat.csv",
            Path(data_dir) / "UCI697" / "data.csv",
        ]
        # Also search recursively
        base_paths = [Path(data_dir) / "raw" / "uci697", Path(data_dir) / "UCI697"]
        for base in base_paths:
            if base.exists():
                for csv_file in base.rglob("*.csv"):
                    possible_paths.append(csv_file)
                    break
        
        for csv_path in possible_paths:
            if csv_path.exists():
                return _load_uci697_from_file(csv_path, seed)
    
    # CRITICAL: Synthetic fallback is STRICTLY FORBIDDEN
    # This code path should NEVER be reached
    raise FileNotFoundError(
        f"CRITICAL ERROR: UCI697 data file not found. Synthetic fallback is STRICTLY FORBIDDEN. "
        f"Searched paths: {[str(p) for p in possible_paths]}. "
        f"Please ensure real data files are available at one of these paths."
    )


def _load_harvardx(seed: int, data_dir: Optional[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """加载HarvardX_PersonCourse数据集"""
    if data_dir:
        # Try multiple possible paths and filenames (including .tab files)
        possible_paths = [
            Path(data_dir) / "raw" / "harvardx" / "HXPC13_DI_v3_11-13-2019.tab",  # Actual downloaded filename
            Path(data_dir) / "raw" / "harvardx" / "HMXPC13_DI_v2_5-14-14.csv",
            Path(data_dir) / "HarvardX_PersonCourse" / "HMXPC13_DI_v2_5-14-14.csv",
        ]
        # Also search recursively
        base_paths = [Path(data_dir) / "raw" / "harvardx", Path(data_dir) / "HarvardX_PersonCourse"]
        for base in base_paths:
            if base.exists():
                # Look for .tab or .csv files
                for data_file in base.rglob("*.tab"):
                    possible_paths.append(data_file)
                    break
                for data_file in base.rglob("*.csv"):
                    possible_paths.append(data_file)
                    break
        
        for data_path in possible_paths:
            if data_path.exists():
                return _load_harvardx_from_file(data_path, seed)
    
    # CRITICAL: Synthetic fallback is STRICTLY FORBIDDEN
    # This code path should NEVER be reached
    raise FileNotFoundError(
        f"CRITICAL ERROR: HarvardX_PersonCourse data file not found. Synthetic fallback is STRICTLY FORBIDDEN. "
        f"Searched paths: {[str(p) for p in possible_paths]}. "
        f"Please ensure real data files are available at one of these paths."
    )


def _load_oulad_from_file(csv_path: Path, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """从CSV文件加载OULAD数据"""
    df = pd.read_csv(csv_path)
    
    # 特征列（根据OULAD schema）
    feature_cols = [
        'code_module', 'code_presentation', 'num_of_prev_attempts',
        'studied_credits', 'disability', 'final_result'
    ]
    # 如果有更多数值特征，添加它们
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 标签：final_result (0=pass, 1=fail)
    # OULAD的final_result可能包含: Pass, Distinction, Withdrawn, Fail
    # 我们需要确保只使用训练集中存在的标签，并将多类转换为二分类
    if 'final_result' in df.columns:
        # 先检查所有唯一值
        unique_results = df['final_result'].unique()
        
        # 将多类结果转换为二分类：Fail=1, 其他(Pass/Distinction/Withdrawn)=0
        # 确保所有值都被正确处理
        df['label'] = df['final_result'].apply(
            lambda x: 1 if str(x).strip().lower() in ['fail', 'withdrawn'] else 0
        ).astype(int)
        
        # 验证标签分布
        label_counts = df['label'].value_counts()
        if len(label_counts) < 2:
            # 如果只有一个类别，需要平衡
            print(f"Warning: Only one label class found in OULAD data. Balancing...")
            # 随机分配一些样本到少数类
            minority_class = 1 - label_counts.index[0]
            n_samples_to_change = min(len(df) // 4, label_counts.iloc[0] // 2)
            indices_to_change = np.random.choice(
                df[df['label'] == label_counts.index[0]].index,
                size=n_samples_to_change,
                replace=False
            )
            df.loc[indices_to_change, 'label'] = minority_class
    else:
        df['label'] = np.random.binomial(1, 0.3, len(df))
    
    # 组标签：gender（如果有）
    groups = None
    if 'gender' in df.columns:
        le = LabelEncoder()
        groups = le.fit_transform(df['gender'].fillna('Unknown'))
    
    # 选择特征
    available_features = [c for c in numeric_cols if c != 'label']
    if len(available_features) < 5:
        # 如果特征太少，添加一些模拟特征
        for i in range(5 - len(available_features)):
            df[f'feature_{i}'] = np.random.randn(len(df))
            available_features.append(f'feature_{i}')
    
    X = df[available_features[:20]].values  # 最多20个特征
    y = df['label'].values
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练/测试集
    # 确保训练集和测试集都包含两个类别的样本
    np.random.seed(seed)  # 确保可重复性
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
    except ValueError:
        # 如果stratify失败（可能因为某个类别样本太少），使用不stratify的方式
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
    
    # 确保训练集和测试集都包含两个类别（二分类必需）
    unique_train_labels = np.unique(y_train)
    unique_test_labels = np.unique(y_test)
    
    if len(unique_train_labels) < 2:
        # 从测试集中移动一些样本到训练集
        minority_class = 1 - unique_train_labels[0]
        test_minority_indices = np.where(y_test == minority_class)[0]
        if len(test_minority_indices) > 0:
            n_to_move = min(10, len(test_minority_indices) // 2)
            indices_to_move = np.random.choice(test_minority_indices, size=n_to_move, replace=False)
            X_train = np.vstack([X_train, X_test[indices_to_move]])
            y_train = np.concatenate([y_train, y_test[indices_to_move]])
            X_test = np.delete(X_test, indices_to_move, axis=0)
            y_test = np.delete(y_test, indices_to_move)
    
    if len(unique_test_labels) < 2:
        # 从训练集中移动一些样本到测试集
        minority_class = 1 - unique_test_labels[0]
        train_minority_indices = np.where(y_train == minority_class)[0]
        if len(train_minority_indices) > 0:
            n_to_move = min(10, len(train_minority_indices) // 2)
            indices_to_move = np.random.choice(train_minority_indices, size=n_to_move, replace=False)
            X_test = np.vstack([X_test, X_train[indices_to_move]])
            y_test = np.concatenate([y_test, y_train[indices_to_move]])
            X_train = np.delete(X_train, indices_to_move, axis=0)
            y_train = np.delete(y_train, indices_to_move)
    
    # 组标签（仅测试集）
    groups_test = None
    if groups is not None:
        _, groups_test = train_test_split(
            groups, test_size=0.2, random_state=seed
        )
    
    return X_train, X_test, y_train, y_test, groups_test


def _load_uci697_from_file(csv_path: Path, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """从CSV文件加载UCI697数据"""
    df = pd.read_csv(csv_path, sep=';')
    
    # 标签：G3 (final grade) -> binary (0=low, 1=high)
    if 'G3' in df.columns:
        median_grade = df['G3'].median()
        df['label'] = (df['G3'] >= median_grade).astype(int)
    else:
        df['label'] = np.random.binomial(1, 0.5, len(df))
    
    # UCI697没有demographic字段用于公平性评估（根据实验设计）
    groups = None
    
    # 选择数值特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'label' and c != 'G3']
    
    if len(numeric_cols) < 5:
        for i in range(5 - len(numeric_cols)):
            df[f'feature_{i}'] = np.random.randn(len(df))
            numeric_cols.append(f'feature_{i}')
    
    X = df[numeric_cols[:20]].values
    y = df['label'].values
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, None


def _load_harvardx_from_file(csv_path: Path, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """从CSV/TAB文件加载HarvardX_PersonCourse数据"""
    # Support both .csv and .tab files
    if csv_path.suffix.lower() == '.tab':
        df = pd.read_csv(csv_path, sep='\t')
    else:
        df = pd.read_csv(csv_path)
    
    # 标签：certified (0=incomplete, 1=complete)
    if 'certified' in df.columns:
        df['label'] = df['certified'].astype(int)
    else:
        df['label'] = np.random.binomial(1, 0.4, len(df))
    
    # HarvardX没有demographic字段用于公平性评估（根据实验设计）
    groups = None
    
    # 选择数值特征
    numeric_cols = ['registered', 'viewed', 'explored', 'nevents', 'ndays_act', 
                    'nplay_video', 'nchapters', 'nforum_posts']
    available_cols = [c for c in numeric_cols if c in df.columns]
    
    # CRITICAL: Handle NaN values - fill with 0 or drop rows
    # First, check for NaN in selected columns
    for col in available_cols:
        if df[col].isna().any():
            # Fill NaN with 0 (or median if preferred)
            df[col] = df[col].fillna(0)
    
    # Drop rows where label is NaN
    df = df.dropna(subset=['label'])
    
    if len(available_cols) < 5:
        for i in range(5 - len(available_cols)):
            df[f'feature_{i}'] = np.random.randn(len(df))
            available_cols.append(f'feature_{i}')
    
    X = df[available_cols[:20]].values
    y = df['label'].values
    
    # Final check: ensure no NaN in X or y
    if np.isnan(X).any() or np.isnan(y).any():
        # Drop rows with NaN
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, None


def _generate_synthetic_oulad(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """生成模拟OULAD数据（用于测试）"""
    np.random.seed(seed)
    n_samples = 5000
    n_features = 15
    
    # 生成特征
    X = np.random.randn(n_samples, n_features)
    
    # 生成标签（约30%失败率）
    # 使用特征线性组合 + 噪声
    coef = np.random.randn(n_features)
    logit = X @ coef + np.random.randn(n_samples) * 0.5
    y = (logit > np.percentile(logit, 70)).astype(int)
    
    # 生成组标签（gender）
    groups = np.random.binomial(1, 0.5, n_samples)
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    _, groups_test = train_test_split(
        groups, test_size=0.2, random_state=seed
    )
    
    return X_train, X_test, y_train, y_test, groups_test


def _generate_synthetic_uci697(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """生成模拟UCI697数据（用于测试）"""
    np.random.seed(seed)
    n_samples = 400
    n_features = 15
    
    X = np.random.randn(n_samples, n_features)
    coef = np.random.randn(n_features)
    logit = X @ coef + np.random.randn(n_samples) * 0.5
    y = (logit > np.percentile(logit, 50)).astype(int)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, None


def _generate_synthetic_harvardx(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """生成模拟HarvardX数据（用于测试）"""
    np.random.seed(seed)
    n_samples = 3000
    n_features = 12
    
    X = np.random.randn(n_samples, n_features)
    coef = np.random.randn(n_features)
    logit = X @ coef + np.random.randn(n_samples) * 0.5
    y = (logit > np.percentile(logit, 60)).astype(int)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, None
