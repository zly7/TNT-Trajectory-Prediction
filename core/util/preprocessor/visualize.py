from core.util.preprocessor.argoverse_preprocess_v2 import ArgoversePreprocessor

if __name__ == "__main__":
    processor = ArgoversePreprocessor(root_dir="/home/zhuhe/Dataset/data")
    processor.visualize_data(data="../../../../Dataset/interm_data_small/train_intermediate/raw/features_211684.pkl")
