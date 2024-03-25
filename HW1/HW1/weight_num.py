import os
folder = "C:/Users/User/Desktop/MLChip/HW1/HW1/data"
data_list = os.listdir(folder)
for i in data_list:
    file_name = f'{folder}/{i}'
    with open(file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        line_count = len(lines)

        print(f"檔案 {file_name} 共有 {line_count} 行。")