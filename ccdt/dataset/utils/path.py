import os


def get_valid_paths(root_dir, file_formats, recursion=True):
    files_name = []
    # 使用系统方法，递归纵向遍历根路径、子路径、文件
    for root, dirs, files in os.walk(root_dir, topdown=False):
        # 对目录和文件进行升序排序
        dirs.sort()
        files.sort()
        # 遍历文件名称列表
        for file in files:
            # 获取文件后缀
            file_suffix = os.path.splitext(file)[-1]
            # 如果读取的文件后缀，在指定的后缀列表中，则返回真继续往下执行
            if file_suffix in file_formats:
                # 如果文件在文件列表中，则返回真继续往下执行
                if file in files:
                    files_name.append(file)
        # 如果递归遍历参数设置为假，默认设置为真传入为假，并且根路径等于传入的根路径，则返回真继续往下执行后终止递归遍历
        if recursion is False and root == root_dir:
            break
    return files_name
