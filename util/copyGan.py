import pandas as pd

# 读取 CSV 文件
df = pd.read_excel("/home/itaer2/zxy/shixi/retfound2/datasets/retina.xlsx")  # 改为你的文件名

# 要匹配的诊断值列表
target_values = [5, 6, 8,9, 11, 12,13]

# 找出 diagnosis1~4 中任一列包含目标值的行
mask = df[['diagnosis1', 'diagnosis2', 'diagnosis3', 'diagnosis4']].isin(target_values).any(axis=1)
df_selected = df[mask].copy()

# 修改 image_name，加上 -stylized-1918
df_selected['image_name'] = df_selected['image_name'].apply(lambda x: x.replace(".jpg", "-stylized-1918.jpg"))

# 合并原始数据和新增 stylized 行
df_augmented = pd.concat([df, df_selected], ignore_index=True)

# 保存为新文件
df_augmented.to_csv("retina_gan2.csv", index=False)
