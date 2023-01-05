# 这个函数方便切换小的pt文件和真正训练的pt文件


import os
if __name__ == "__main__":
    path1 = "../Dataset/interm_data/train_intermediate/processed"
    path2 = "../Dataset/interm_data/train_intermediate/processed_small"
    path3 = "../Dataset/interm_data/train_intermediate/processed_origin"
    path4 = "../Dataset/interm_data/train_intermediate/processed_middle"

    which_change_to = path4  # 需要手动更改目标

    if not os.path.exists(path2):
        os.rename(path1,path2)
    elif not os.path.exists(path3):
        os.rename(path1,path3)
    elif not os.path.exists(path4):
        os.rename(path1,path4)

    os.rename(which_change_to,path1)

    path1 = "../Dataset/interm_data/val_intermediate/processed"
    path2 = "../Dataset/interm_data/val_intermediate/processed_small"
    path3 = "../Dataset/interm_data/val_intermediate/processed_origin"
    path4 = "../Dataset/interm_data/val_intermediate/processed_middle"

    which_change_to = path4  # 需要手动更改目标

    if not os.path.exists(path2):
        os.rename(path1,path2)
    elif not os.path.exists(path3):
        os.rename(path1,path3)
    elif not os.path.exists(path4):
        os.rename(path1,path4)

    os.rename(which_change_to,path1)
    

    # if os.path.exists(path2):
    #     os.rename(path1,path3)
    #     os.rename(path2,path1)
    #     print("把小数据集替换到实际使用的数据集")
    # else:
    #     os.rename(path1,path2)
    #     os.rename(path3,path1)
    #     print("把原数据集替换到实际使用的数据集")
    
    # path1 = "../Dataset/interm_data/val_intermediate/processed"
    # path2 = "../Dataset/interm_data/val_intermediate/processed_small"
    # path3 = "../Dataset/interm_data/val_intermediate/processed_origin"

    # if os.path.exists(path2):
    #     os.rename(path1,path3)
    #     os.rename(path2,path1)
    #     print("把小数据集替换到实际使用的数据集")
    # else:
    #     os.rename(path1,path2)
    #     os.rename(path3,path1)
    #     print("把原数据集替换到实际使用的数据集")
