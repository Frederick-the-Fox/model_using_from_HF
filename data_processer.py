import xlrd
import json
 
excel_path = "/data/wangyuanchun/NLP_course/dataset/国际疾病分类 ICD-10北京临床版v601.xlsx"

excel = xlrd.open_workbook(excel_path,encoding_override="utf-8")

#获取sheet对象
sheet = excel.sheets()[0]
 
sheet_row_mount = sheet.nrows
sheet_col_mount = sheet.ncols
 
print("row number: {0} ; col number: {1}".format(sheet_row_mount, sheet_col_mount))

item_list = []

for x in range(0, sheet_row_mount):
    y = 1
    item_list.append(sheet.cell_value(x, y))

# print(len(item_list))

with open('../dataset/dev.json', 'r') as file_src:
    data_src = json.load(file_src)
file_src.close()

# print(data_src[0].items())

data_json_list = []
ite = 0
for each_json in data_src:
    print('this is ite : {}'.format(ite))
    classtxt = each_json['normalized_result']
    class_list = classtxt.split('##')
    # print (class_list)
    iter = 0
    for each_class in class_list:
        label_list = [[0, 1] for x in range(0, 40474)]
        try:
            # label_list.append(item_list.index(each_class))
            label_list[item_list.index(each_class)] = [1, 0]
        except:
            continue
    target_json = {}
    target_json["text"] = each_json['text']
    target_json["label"] = label_list
    data_json_list.append(target_json)
    ite = ite + 1

# js_str = json.dumps(data_json_list, ensure_ascii=False, indent=4)
# with open('../dataset/post_processed/train_mse_2.json', 'w') as output:
#     output.write(js_str)
# output.close()

# js_str = json.dumps(data_json_list, ensure_ascii=False, indent=4)
with open('../dataset/post_processed/val_mse_2_line.json', 'w') as output:
    output.write('[')
    flag = 0
    for each in data_json_list:
        if flag == 1:
            output.write(',')
        js_str = json.dumps(each, ensure_ascii=False)
        output.write(js_str)
        output.write('\n')
        flag = 1
    output.write(']')
output.close()
# with open('../dataset/post_processed/train_mse.json', 'a') as output:
#     json.dump(data_json_list, output, ensure_ascii=False)
#     output.write('\n')