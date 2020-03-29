import os

cwd = os.getcwd()
common_dir = os.path.dirname(cwd)
visdrone_dataset_dir = os.path.join(common_dir, 'Datasets', 'VisDrone')

expr_dir = os.path.join(cwd, 'experiments')
model_base = os.path.join(cwd, 'tracking/model_data')

det_train_data_dir = os.path.join(visdrone_dataset_dir, 'VisDrone2019-DET-train')
det_val_data_dir = os.path.join(visdrone_dataset_dir, 'VisDrone2019-DET-val')
det_train_img_dir = os.path.join(det_train_data_dir, 'images')
det_train_anno_dir = os.path.join(det_train_data_dir, 'annotations')
det_val_img_dir = os.path.join(det_val_data_dir, 'images')
det_val_anno_dir = os.path.join(det_val_data_dir, 'annotations')

mot_train_data_dir = os.path.join(visdrone_dataset_dir, 'VisDrone2019-VID_MOT-train')
mot_val_data_dir = os.path.join(visdrone_dataset_dir, 'VisDrone2019-VID_MOT-val')
mot_train_seq_dir = os.path.join(mot_train_data_dir, 'sequences')
mot_train_anno_dir = os.path.join(mot_train_data_dir, 'annotations')
mot_val_seq_dir = os.path.join(mot_val_data_dir, 'sequences')
mot_val_anno_dir = os.path.join(mot_val_data_dir, 'annotations')
