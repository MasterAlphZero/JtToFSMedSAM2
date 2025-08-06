import os
import csv
import time  # 添加时间模块用于生成唯一文件名
from datetime import datetime  # 添加日期时间模块
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["JITTOR_ENABLE_MPS_FALLBACK"] = "0"
import numpy as np
import jittor as jt
from utils.npz_loader import group_sup_img_parts, normalize_img

# saved_npz_path = '/root/autodl-tmp/JtToFSMedSAM2/FS_MedSAM2_data/saved_npz_labels16'
saved_npz_path = '/root/autodl-tmp/JtToFSMedSAM2/FS_MedSAM2/example_data'
ckpt_path = '/root/autodl-tmp/JtToFSMedSAM2/SAM2/checkpoints'

def calculate_dice_compo(prediction, groundtruth):
    # 计算TP、FP、FN
    tp = np.sum((prediction == 1) & (groundtruth == 1))
    fp = np.sum((prediction == 1) & (groundtruth == 0))
    fn = np.sum((prediction == 0) & (groundtruth == 1))
    return tp, fp, fn


# select the device for computation
if jt.compiler.has_cuda:
    device = "cuda"
else:
    device = "cpu"
print(f"using device: {device}")

if device == "cuda":
    jt.flags.use_cuda = 1

from sam2.build_fsmedsam2 import build_fsmedsam2_video_predictor


data_dict = {}
for i, group in enumerate(group_sup_img_parts(saved_npz_path)):
    exp_data = np.load(os.path.join(saved_npz_path, group['files'][0]))
    case_id = group['case_id']
    if case_id not in data_dict:
        data_dict[case_id] = []
    sup_img_part = exp_data['sup_img_part']
    sup_img_part = normalize_img(sup_img_part)
    sup_fgm_part = exp_data['sup_fgm_part']
    label_id = exp_data['labels'][0]

    infer_nums = len(group['files'])
    query_imgs = []
    query_labels = []
    query_names = group['files']

    for infer_id in range(infer_nums):
        cur_image = np.load(os.path.join(saved_npz_path, group['files'][infer_id]))['query_images']
        cur_image = normalize_img(cur_image)
        cur_label = np.load(os.path.join(saved_npz_path,group['files'][infer_id]))['query_labels']
        query_imgs.append(cur_image)
        query_labels.append(cur_label)

    data_dict[case_id].append({'sup_img': sup_img_part, 'sup_label': sup_fgm_part, 'label_id': label_id, 'query_imgs': query_imgs, 'query_labels': query_labels,'query_names':query_names})
    
model_type_dict = {'sam2_hiera_t':'sam2_hiera_tiny', 'sam2_hiera_s':'sam2_hiera_small', 'sam2_hiera_b+':'sam2_hiera_base_plus','sam2_hiera_l':'sam2_hiera_large'}
# 创建CSV日志文件（带时间戳确保唯一性）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"inference_results_{timestamp}.csv"

# 写入CSV表头
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['model_cfg', 'label_id', 'tp', 'fp', 'fn', 'dice'])

for k,v in model_type_dict.items():
    metric_dict = {}
    sam2_checkpoint = os.path.join(ckpt_path, f'{v}.pt')
    model_cfg = f"{k}.yaml"

    predictor = build_fsmedsam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    for case_id in data_dict:
        data_list = data_dict[case_id]
        sup_img_list = []
        sup_label_list = []
        query_imgs_list = []
        query_labels_list = []
        query_names_list = []
        label_id = data_list[0]['label_id']
        for sub_volume in data_list:
            sup_img_list.append(sub_volume['sup_img'])
            sup_label_list.append(sub_volume['sup_label'])
            query_imgs_list.extend(sub_volume['query_imgs'])
            query_labels_list.extend(sub_volume['query_labels'])
            query_names_list.extend(sub_volume['query_names'])
        combined = list(zip(query_imgs_list, query_labels_list, query_names_list))
        combined_sorted = sorted(combined, key=lambda x: int(x[2].split('_z')[1].split('.')[0]))
        query_imgs_list, query_labels_list, query_names_list = zip(*combined_sorted)
        query_imgs_list, query_labels_list, query_names_list = list(query_imgs_list), list(query_labels_list), list(query_names_list)
        
        video_segments = np.zeros((len(query_imgs_list), query_imgs_list[0].shape[-2], query_imgs_list[0].shape[-1]), dtype=np.uint8)
        all_labels = np.concatenate(query_labels_list,axis=0)
        for i in range(len(query_imgs_list)):
            all_images = np.concatenate(sup_img_list+[query_imgs_list[i]],axis=0)
            inference_state = predictor.init_state_by_np_data(images_np=all_images)
            predictor.reset_state(inference_state)
            labels = np.array([1], np.int32)
            for j in range(len(sup_img_list)):
                 _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=j,
                    obj_id=1,
                    mask=sup_label_list[j][0],
                )
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=len(sup_img_list)):
                video_segments[i] = (out_mask_logits[0] > 0.0).cpu().numpy()
        

        all_vis = []
        for i in range(len(query_imgs_list)):
            query_pred = video_segments[i].squeeze()
            query_labels = all_labels[i].squeeze()
            # query_img = normalize_img(all_images[j])*255
            # query_img = query_img.transpose(1, 2, 0).astype(np.uint8)
            # vis_img1 = query_img.copy()
            # vis_img2 = query_img.copy()
            # vis_img3 = query_img.copy()

            # vis_img2[query_pred>0] = (0,255,255)
            # vis_img3[query_labels>0] = (255,0,255)
            # all_vis.append(np.hstack([vis_img1, vis_img2, vis_img3]))
            tp, fp, fn = calculate_dice_compo(query_pred, query_labels)
            
            if label_id not in metric_dict:
                metric_dict[label_id] = {"tp": 0, "fp": 0, "fn": 0}
            metric_dict[label_id]["tp"] += tp
            metric_dict[label_id]["fp"] += fp
            metric_dict[label_id]["fn"] += fn
        # concat_img = np.vstack(all_vis)
        # cv2.imwrite(f'{save_dir}/case_{case_id}.jpg', concat_img)

        # slice_dice = metric.dc(query_pred, query_labels)
        # if labels_id[0] not in metric_dict:
        #     metric_dict[labels_id[0]] = []
        # metric_dict[labels_id[0]].append(slice_dice)
    # 将结果写入CSV文件
    with open(csv_filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for k, v in metric_dict.items():
            dice_score = 2 * v['tp'] / (2 * v['tp'] + v['fp'] + v['fn'])
            print(model_cfg)
            print(f"label: {k}, dice: {dice_score}")
            # 写入模型配置、标签ID、组件值和Dice分数
            csv_writer.writerow([model_cfg, k, v['tp'], v['fp'], v['fn'], dice_score])


