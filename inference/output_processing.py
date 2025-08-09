# ------------------------------------------------------------- #
# End-to-end one-stage with global and local information        #
# Process raw model output                                      #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #


import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn.functional as F

from config.config import MODEL_CONFIG, TEST_CONFIG


class OutputProcessing():
    def __init__(self, dataset_config, device):
        self.dataset_config = dataset_config
        self.device = device


    # Non maximum suppression using junction location
    #   Sort the junction acording to confidence score
    #   Pick the junction with highest confidence
    #   Delete others whose location is within 32 pixels with the selected one
    def nms(self, junctions):
        new_junctions = [junction for junction in junctions]
        if len(new_junctions) > 0:
            new_junctions = sorted(new_junctions, key=lambda x: x[0], reverse=True)

        nms_junctions = []
        while new_junctions:
            selected_junction = new_junctions.pop(0)

            new_junctions = [
                junction
                for junction in new_junctions
                if torch.norm(torch.tensor((junction[1] - selected_junction[1], junction[2] - selected_junction[2]))) > TEST_CONFIG['nms_thres']
            ]

            nms_junctions.append(selected_junction)

        if len(nms_junctions) > 0:
            nms_junctions = torch.stack(nms_junctions, dim=0)

        return nms_junctions
    


    # Calculating the slot prediction from model outputs
    def process_output(self, output):
        ## Get junction prediction 
        h, w, c = output.shape
        junctions = output.reshape(-1, c)
        junctions = junctions[:, 9:]
        indices = torch.arange(len(junctions)).reshape(-1, 1)
        xid, yid = torch.unravel_index(indices, (h, w))
        cell_id = torch.cat([yid, xid], dim=-1).to(self.device)
        # Remove predictions with low confidence
        indices = junctions[:, 0] > TEST_CONFIG['junc_thres']
        junctions = junctions[indices]
        cell_id = cell_id[indices]
        
        # Calculate junction coordinate
        junctions[:, 1:3] = ((cell_id + 0.5) + (junctions[:, 1:3] - 0.5)) * MODEL_CONFIG['model_stride']
        junctions[:, 3:5] = F.normalize(junctions[:, 3:5], dim=-1)
        # NMS
        junctions = self.nms(junctions)
        if len(junctions) == 0:
            return np.array([])

        junctions = junctions[:, 1:]



        ## Get raw slot prediction from global information output
        h, w, c = output.shape
        raw_slots = output.reshape((-1, c))
        raw_slots = raw_slots[:, :9] 
        indices = torch.arange(len(raw_slots)).reshape(-1, 1)
        xid, yid = torch.unravel_index(indices, (h, w))
        cell_id = torch.cat([yid, xid], dim=-1).to(self.device)
        
        # Remove predictions with low confidence 
        indices = raw_slots[:, 0] > TEST_CONFIG['inslot_thres']
        raw_slots = raw_slots[indices]
        cell_id = cell_id[indices]
        
        # Change slot values to correct range
        if len(raw_slots) > 0:
            # Rough junction location
            raw_slots[:, 1:3] = (cell_id + 0.5) * MODEL_CONFIG['model_stride'] + raw_slots[:, 1:3] * self.dataset_config['l_max'] 
            raw_slots[:, 3:5] = (cell_id + 0.5) * MODEL_CONFIG['model_stride'] + raw_slots[:, 3:5] * self.dataset_config['l_max'] 

            slot_type = torch.argmax(raw_slots[:, 5:8], dim=-1)
            slot_occ = (raw_slots[:, 8] > 0.5).to(torch.float32)
        else:
            return np.array([])

        
        # Replace the rough junction in raw_slots with the precise junction if the distance between those two is less than threshold
        junc1 = raw_slots[:, 1:3]
        dist1 = torch.cdist(junc1, junctions[:, 0:2])
        min_dist1, indices1 = torch.min(dist1, dim=-1)

        junc2 = raw_slots[:, 3:5]
        dist2 = torch.cdist(junc2, junctions[:, 0:2])
        min_dist2, indices2 = torch.min(dist2, dim=-1)

        check = torch.logical_and((min_dist1 < TEST_CONFIG['dist_thres']), (min_dist2 < TEST_CONFIG['dist_thres']))

        if torch.sum(check) > 0:
            final_slots = torch.cat([junctions[indices1[check]][:, 0:2],
                                     junctions[indices2[check]][:, 0:2],
                                     junctions[indices1[check]][:, 2:4],
                                     junctions[indices2[check]][:, 2:4],
                                     slot_type[check].reshape(-1, 1), 
                                     slot_occ[check].reshape(-1, 1)], dim=-1)
            final_slots = torch.unique(final_slots, dim=0)
            final_slots = final_slots.cpu().numpy()
        else:
            final_slots = np.array([])
        
        return final_slots
            

    

    def visualize_preds(self, imgs, final_slots, index):
        img = imgs[index].detach().cpu().permute(1, 2, 0)
        plt.figure()
        plt.axis('off')
        plt.imshow(img)

        for item in final_slots:
            junc1 = np.array([item[0], item[1]])
            junc2 = np.array([item[2], item[3]])

            ori1 = np.array([junc1[0] + item[4]*25, junc1[1] + item[5]*25])
            ori2 = np.array([junc2[0] + item[6]*25, junc2[1] + item[7]*25])

            slot_type = int(item[8])
            slot_occ = item[9]
            if slot_occ == 0:
                plt.plot([junc1[0], junc2[0]], [junc1[1], junc2[1]], '-', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
                plt.plot([junc1[0], ori1[0]], [junc1[1], ori1[1]], '-', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
                plt.plot([junc2[0], ori2[0]], [junc2[1], ori2[1]], '-', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
            else:
                plt.plot([junc1[0], junc2[0]], [junc1[1], junc2[1]], '--', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
                plt.plot([junc1[0], ori1[0]], [junc1[1], ori1[1]], '--', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
                plt.plot([junc2[0], ori2[0]], [junc2[1], ori2[1]], '--', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
        
        # plt.savefig('b.png')
        # plt.close()
        plt.show()


    