# ------------------------------------------------------------- #
# End-to-end one-stage with global and local information        #
# Configiguration file                                          #
#       - Dataset configurations (SNU, PS2.0)                   #
#       - Model configurations                                  #
#       - Training/Testing set up                               #
#       - Evaluation criteria                                   #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #


WEIGHT_DIR = './weights/'
LOG_DIR = './log/'


DATASET = 'snu'
DATASET_CONFIG = {
    'ps2': {
        'dataset_path': '/home/ivpg/HUY/Dataset/PS2/',
        'ori_height': 600,
        'ori_width': 600,
        'img_height': 416,
        'img_width': 416,
        'ratio': 416/600, # img_height / ori_height
        'l_max': 7.0 * (416 / 10.0), # 7m * (img_height / 10m)
        'l_mean': [200, 100, 200], # per, par, sla

        'class_name': ['perpendicular', 'parallel', 'slanted'],
        'color_dict': ['g', 'r', 'b'],
        'class_weights': [1.6718, 2.7136, 29.9873], #[per: 5668, par: 3492, sla: 316]
        'occupancy_weights': [1, 1],
        'flip_horizontal': True,    

        'inslot_weights': [1.1007, 10.9307], #[not in (0): 1508828, in (1): 151935]
        'junc_weights': [1.0057, 175.2599], #[not contain (0): 1651287, contain (1): 9476]

    },
    'snu': { 
        'dataset_path': '/home/ivpg/HUY/Dataset/SNU/',
        'ori_height': 768,
        'ori_width': 256,
        'img_height': 576,
        'img_width': 192,
        'ratio': 576/768, # img_height / ori_height
        'l_max': 400 * (576/768), # 400pixel * ratio
        'l_mean': [75, 200, 200], # par, per, sla

        'conf_weights': [1.1171, 9.5427], # [not contain (0): 442298, contain (1): 51775]
        'class_name': ['parallel', 'perpendicular', 'slanted'],
        'color_dict': ['r', 'g', 'b'],
        'class_weights': [8.2065, 1.2331, 14.8779], #[par: 6309, per: 41986, sla: 3480]
        'occupancy_weights': [1.7136, 2.4013], #[vac (0): 30214, occ (1): 21561]
        'flip_horizontal': False,

        'inslot_weights': [1.1726, 6.7925], #[not in (0): 1685340, in (1): 290952]
        'junc_weights': [1.0269, 38.1707], #[not contain (0): 1924517, contain (1): 51775]

    }
}



MODEL_CONFIG = {
    'd_model': 256,
    'model_stride': 32,
    'label_channel': 14,
}




LOSS_CONFIG = {
    'inside_slot': 100,
    'junc': 500,
    'slot_type': 1,
    'slot_occ': 50,
    'junc_conf': 500,
    'junc_loc': 5000,
    'junc_ori': 1000,
}



TRAIN_CONFIG = {
    'batch_size': 32,
    'epoch_num': 100,
    'lr_init': 1e-4,
    'lr_end': 1e-6,
}


TEST_CONFIG = {
    'batch_size': 1,
    'weight_path': './weights/snu_20250808_212458/model_weights.pth',
    'evaluation_mode': 'loose',

    'junc_thres': 0.5,
    'inslot_thres': 0.01,
    'nms_thres': 32, # pixel

    'dist_thres': 12, # pixel

    'loose_loc_thres': 12, 
    'loose_ori_thres': 10,
    'tight_loc_thres': 6,
    'tight_ori_thres': 5,
}

