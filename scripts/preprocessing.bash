#export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# python core/util/preprocessor/argoverse_preprocess_v2.py --root dataset/ --dest dataset

# generate a small subset to test the training program
/home/zhuhe/anaconda3/envs/pytorch/bin/python3 /home/zhuhe/TNT-Trajectory-Prediction/core/util/preprocessor/argoverse_preprocess_v2.py --root /home/zhuhe/Dataset/ --dest dataset -s
