import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

def count_labels(df, class_num=15, name='', save_path=None):
    """
    统计多标签中每个类别出现的次数，并输出和保存。
    """
    counter = Counter()
    for label_str in df['labels']:
        labels = label_str.split(',')
        counter.update(int(l) for l in labels if l != '')

    # 打印统计信息
    print(f"\n{name} 集中每个类别的数量：")
    for i in range(class_num):
        print(f"  类别 {i}: {counter[i]}")

    # 保存为 CSV 文件
    if save_path:
        stats = {'class': list(range(class_num)), 'count': [counter[i] for i in range(class_num)]}
        df_stats = pd.DataFrame(stats)
        df_stats.to_csv(save_path, index=False)
        print(f"  -> 类别统计信息已保存至 {save_path}")

def convert_and_split(
    excel_path,
    image_root,
    save_dir,
    test_size=0.1,
    val_size=0.1,
    random_state=42,
    class_num=15,
):
    # 1. 读取 Excel 文件
    df = pd.read_csv(excel_path)
    print(f"读取到 {len(df)} 条记录")

    # 2. 整合标签
    def merge_labels(row):
        labels = [row[f'diagnosis{i}'] for i in range(1, 5) if pd.notna(row[f'diagnosis{i}'])]
        return ','.join(str(int(l)) for l in labels if l != '')

    df['labels'] = df.apply(merge_labels, axis=1)

    # 3. 添加完整图像路径
    df['image'] = df['image_name'].apply(lambda x: os.path.join(image_root, x))

    # 4. 只保留我们需要的两列
    df = df[['image', 'labels']]

    # 5. 训练/验证/测试划分
    df_temp, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    val_ratio = val_size / (1 - test_size)
    df_train, df_val = train_test_split(df_temp, test_size=val_ratio, random_state=random_state)

    print(f"\n划分完成：Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 6. 输出并保存每个类别的数量统计
    count_labels(df, class_num=class_num, name='全部', save_path=os.path.join(save_dir, 'count_all.csv'))
    count_labels(df_train, class_num=class_num, name='Train', save_path=os.path.join(save_dir, 'count_train.csv'))
    count_labels(df_val, class_num=class_num, name='Val', save_path=os.path.join(save_dir, 'count_val.csv'))
    count_labels(df_test, class_num=class_num, name='Test', save_path=os.path.join(save_dir, 'count_test.csv'))

    # 7. 保存划分后的 CSV 文件
    df_train.to_csv(os.path.join(save_dir, 'train.csv'), index=False)
    df_val.to_csv(os.path.join(save_dir, 'val.csv'), index=False)
    df_test.to_csv(os.path.join(save_dir, 'test.csv'), index=False)
    print(f"\nCSV 文件保存至 {save_dir}")

# ====== 示例调用 ======
convert_and_split(
    excel_path='/home/itaer2/zxy/shixi/retfound2/datasets/mutil/gan2/retina_gan2.csv',
    image_root='/home/itaer2/zxy/shixi/retfound2/datasets/Main-gan',
    save_dir='/home/itaer2/zxy/shixi/retfound2/datasets/mutil/gan2',
    test_size=0.1,
    val_size=0.1,
    class_num=15
)
