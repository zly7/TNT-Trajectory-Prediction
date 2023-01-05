#!/bin/bash
#SBATCH -o job.%j.out          # 脚本执行的输出将被保存在当job.%j.out文件下，%j表示作业号;
#SBATCH --partition=gpulab02      # 作业提交的指定分区队列为titan
#SBATCH --qos=gpulab02            # 指定作业的QOS
#SBATCH -J processCPU       # 作业在调度系统中的作业名为myFirstJob;
#SBATCH --nodes=1              # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --ntasks-per-node=24   # 每个节点上运行一个任务，默认一情况下也可理解为每个节点使用一个核心；


python /home/songx_lab/cse12012530/TNT-Trajectory-Prediction/core/util/preprocessor/argoverse_preprocess_v2.py