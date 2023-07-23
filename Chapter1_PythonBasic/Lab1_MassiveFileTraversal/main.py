import os
import zipfile


def unzip_data(des_path: str, src_path: str) -> bool:
    '''将src_path路径对应的zip文件解压至新创建的des_path目录'''
    if not os.path.isdir(des_path):
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=des_path)
        z.close()
        return True
    return False


def merge_dict_by_add(des_dict: dict, src_dict: dict) -> dict:
    '''将src_dict中的键值对按照键通过加法合并至des_dict'''
    for key, value in src_dict.items():
        if des_dict.get(key) is not None:
            des_dict[key] += value
        else:
            des_dict[key] = value
    return des_dict


def get_file_type_size(path: str) -> list:
    '''
    统计path目录下文件的类型和大小

    Dependency: merge_dict_by_add
    '''
    type_dict = {}
    size_dict = {}
    files = os.listdir(path)
    for filename in files:
        temp_path = os.path.join(path, filename)
        if os.path.isdir(temp_path):
            # 递归解析子目录
            type_size_result = get_file_type_size(temp_path)
            merge_dict_by_add(type_dict, type_size_result[0])
            merge_dict_by_add(size_dict, type_size_result[1])
        elif os.path.isfile(temp_path):
            # 获取文件后缀
            type_name = os.path.splitext(temp_path)[1]
            if not type_name:
                # 无后缀名文件
                type_dict.setdefault("None", 0)
                type_dict["None"] += 1
                size_dict.setdefault("None", 0)
                size_dict["None"] += os.path.getsize(temp_path)
            else:
                # 有后缀名文件
                type_dict.setdefault(type_name, 0)
                type_dict[type_name] += 1
                size_dict.setdefault(type_name, 0)
                size_dict[type_name] += os.path.getsize(temp_path)
    return [type_dict, size_dict]


if __name__ == '__main__':
    # res = unzip_data(
    #     r'./Chapter1_PythonBasic/1.zip',
    #     r'./Chapter1_PythonBasic/Lab1_MassiveFileTraversal/1',
    # )
    # print(res)
    [type_dict, size_dict] = get_file_type_size(r"./")
    print(type_dict)
    print(size_dict)
