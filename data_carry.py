import  os
import shutil
if __name__ == "__main__":
    root_dir = "/home/zhuhe/Dataset/data/train"
    root_dir_to = "/home/zhuhe/Dataset/data_carry/train"
    temp_list = os.listdir(root_dir)
    temp_list1 = sorted(temp_list)[:1000]
    temp_list2 = sorted(temp_list)[2501:3000]
    # for t_path in temp_list1 :
    #     fin_path = os.path.join(root_dir_to,t_path)
    #     os.remove(fin_path)
    for t_path in temp_list1:
        fin_path = os.path.join(root_dir,t_path)
        to_path = os.path.join(root_dir_to,t_path)
        shutil.copy(fin_path,to_path)



