import os
import csv
import time  # 添加时间模块用于生成唯一文件名
from datetime import datetime  # 添加日期时间模块
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import numpy as np
import torch
import matplotlib.pyplot as plt
import pdb
from utils.npz_loader import npz_loader, group_sup_img_parts, normalize_img
import sys
from PIL import Image
import cv2
saved_npz_path = '/root/autodl-tmp/JtToFSMedSAM2/FS_MedSAM2_data/saved_npz_labels14'
ckpt_path = '/root/autodl-tmp/JtToFSMedSAM2/SAM2/checkpoints'
# 创建可视化输出目录
visualization_dir = os.path.join(os.path.dirname(__file__), "visualization_results")
os.makedirs(visualization_dir, exist_ok=True)
def calculate_dice_compo(prediction, groundtruth):

    tp = np.sum((prediction == 1) & (groundtruth == 1))
    fp = np.sum((prediction == 1) & (groundtruth == 0))
    fn = np.sum((prediction == 0) & (groundtruth == 1))
    return tp, fp, fn


def visualize_slice(original_img, prediction, groundtruth):
    """
    可视化原始图像、预测结果和真实标签
    修复通道处理问题
    """
    # 确保预测和标签是二维的
    prediction = prediction.squeeze()
    groundtruth = groundtruth.squeeze()

    # 处理原始图像的各种可能格式
    if original_img.ndim == 3:
        # 通道在前的格式 (C, H, W) -> (H, W, C)
        if original_img.shape[0] in [1, 3]:
            original_img = np.transpose(original_img, (1, 2, 0))
        # 通道在后的格式 (H, W, C)
        elif original_img.shape[2] in [1, 3]:
            pass
    elif original_img.ndim == 2:
        # 单通道图像，添加通道维度
        original_img = np.expand_dims(original_img, axis=-1)

    # 如果图像是单通道，转换为三通道
    if original_img.shape[-1] == 1:
        original_img = np.concatenate([original_img] * 3, axis=-1)

    # 归一化并转换为8位图像
    if original_img.max() > 1:
        original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
    original_img_8bit = (original_img * 255).astype(np.uint8)

    # 确保所有图像都是三维的 (H, W, C)
    if original_img_8bit.ndim == 2:
        original_img_8bit = np.stack([original_img_8bit] * 3, axis=-1)

    # 创建彩色可视化
    img_original = original_img_8bit.copy()
    img_pred = original_img_8bit.copy()
    img_truth = original_img_8bit.copy()
    img_overlay = original_img_8bit.copy()

    # 用黄色标记预测结果 (BGR)
    if img_pred.shape[-1] == 3:  # 确保是三通道
        # 创建预测掩模的布尔索引
        pred_mask = prediction > 0
        # 只修改掩模区域
        img_pred[pred_mask] = [0, 255, 255]

    # 用紫色标记真实标签 (BGR)
    if img_truth.shape[-1] == 3:  # 确保是三通道
        truth_mask = groundtruth > 0
        img_truth[truth_mask] = [255, 0, 255]

    # 在混合图像中用不同颜色标记预测和真实标签
    if img_overlay.shape[-1] == 3:  # 确保是三通道
        overlay_mask_pred = prediction > 0
        overlay_mask_truth = groundtruth > 0
        img_overlay[overlay_mask_pred] = [0, 255, 255]  # 黄色表示预测
        img_overlay[overlay_mask_truth] = [255, 0, 255]  # 紫色表示真实标签

    # 创建对比图：原始 | 预测 | 真实 | 叠加
    comparison = np.hstack([
        img_original,
        img_pred,
        img_truth,
        img_overlay
    ])

    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5 if comparison.shape[1] > 1000 else 0.7
    thickness = 1
    color = (255, 255, 255)

    # 计算每个子图的起始位置
    w = original_img_8bit.shape[1]
    cv2.putText(comparison, "Original", (10, 30), font, scale, color, thickness)
    cv2.putText(comparison, "Prediction", (w + 10, 30), font, scale, color, thickness)
    cv2.putText(comparison, "Ground Truth", (2 * w + 10, 30), font, scale, color, thickness)
    cv2.putText(comparison, "Overlay", (3 * w + 10, 30), font, scale, color, thickness)

    return comparison


def create_volume_montage(slice_images, cols=5, padding=10):
    """
    创建整个体积的蒙太奇图像
    """
    # 获取单个切片的尺寸
    slice_h, slice_w = slice_images[0].shape[:2]

    # 计算需要多少行
    rows = (len(slice_images) + cols - 1) // cols

    # 创建大画布
    montage_h = rows * slice_h + (rows + 1) * padding
    montage_w = cols * slice_w + (cols + 1) * padding
    montage = np.ones((montage_h, montage_w, 3), dtype=np.uint8) * 50  # 灰色背景

    # 放置每个切片
    for i, img in enumerate(slice_images):
        row = i // cols
        col = i % cols

        # 计算位置
        y_start = padding + row * (slice_h + padding)
        y_end = y_start + slice_h
        x_start = padding + col * (slice_w + padding)
        x_end = x_start + slice_w

        # 放置图像
        montage[y_start:y_end, x_start:x_end] = img

        # 添加切片编号
        cv2.putText(montage, f"Slice {i + 1}",
                    (x_start + 10, y_start + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return montage

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

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
metrics_csv = os.path.join(visualization_dir, f"inference_metrics_{timestamp}.csv")
timing_csv = os.path.join(visualization_dir, f"inference_timing_{timestamp}.csv")

# 写入CSV表头
with open(metrics_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['model_cfg', 'label_id', 'tp', 'fp', 'fn', 'dice'])

with open(timing_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['model_cfg', 'case_id', 'num_slices', 'total_time', 'avg_time_per_slice', 'speed_fps'])
    
for k,v in model_type_dict.items():
    metric_dict = {}
    timing_dict = {}
    sam2_checkpoint = os.path.join(ckpt_path, f'{v}.pt')
    model_cfg = f"{k}.yaml"

    # 为当前模型创建可视化目录
    model_vis_dir = os.path.join(visualization_dir, k)
    os.makedirs(model_vis_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Loading model: {model_cfg}...")
    start_load = time.time()
    predictor = build_fsmedsam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds")

    for case_id in data_dict:
        print(f"\nProcessing case {case_id}...")
        case_start_time = time.time()
        case_vis_dir = os.path.join(model_vis_dir, f"case_{case_id}")
        os.makedirs(case_vis_dir, exist_ok=True)

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
        sup_img_list, sup_label_list = [sup_img_list[1]], [sup_label_list[1]]
        all_images = np.concatenate(sup_img_list+query_imgs_list,axis=0)
        all_labels = np.concatenate(sup_label_list+query_labels_list,axis=0)
        
        init_start = time.time()
        inference_state = predictor.init_state_by_np_data(images_np=all_images)
        predictor.reset_state(inference_state)
        init_time = time.time() - init_start

        labels = np.array([1], np.int32)
        support_time = 0
        for i in range(len(sup_img_list)):
            sup_start = time.time()
            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=i,
                obj_id=1,
                mask=sup_label_list[i][0],
            )
            support_time += time.time() - sup_start
        
        video_segments = np.zeros((all_images.shape[0], all_images.shape[-2], all_images.shape[-1]), dtype=np.uint8)

        # 传播分割结果
        print(f"Propagating segmentation for {len(query_imgs_list)} slices...")
        propagation_times = []
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,start_frame_idx=len(sup_img_list)):
            frame_start = time.time()
            video_segments[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()
            propagation_times.append(time.time() - frame_start)
        # 计算时间指标
        total_propagation_time = sum(propagation_times)
        num_volumes = len(query_imgs_list)
        avg_time_per_slice = total_propagation_time / num_volumes if num_volumes > 0 else 0
        speed_fps = num_volumes / total_propagation_time if total_propagation_time > 0 else 0
        # 记录总时间
        total_case_time = time.time() - case_start_time
        print(f"Case processed in {total_case_time:.2f}s | "
              f"Avg time/slice: {avg_time_per_slice * 1000:.1f}ms | "
              f"Speed: {speed_fps:.1f} fps")
        # 保存时间信息
        timing_dict[case_id] = {
            'num_slices': num_volumes,
            'total_time': total_propagation_time,
            'avg_time_per_slice': avg_time_per_slice,
            'speed_fps': speed_fps
        }
        slice_visualizations = []
        all_vis = []
        for j in range(len(sup_img_list), len(all_images)):
            query_pred = video_segments[j].squeeze()
            query_labels = all_labels[j].squeeze()
            query_img = all_images[j].squeeze()
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
            # 可视化结果
            slice_idx = j - len(sup_img_list)
            slice_name = os.path.splitext(query_names_list[slice_idx])[0]
            # 可视化切片并保存
            slice_img = visualize_slice(query_img, query_pred, query_labels)
            slice_save_path = os.path.join(case_vis_dir, f"{slice_name}.png")
            cv2.imwrite(slice_save_path, slice_img)

            # 收集切片可视化结果用于创建体积蒙太奇
            slice_visualizations.append(slice_img)
        # 创建整个体积的蒙太奇图像
        if slice_visualizations:
            volume_montage = create_volume_montage(slice_visualizations, cols=min(5, len(slice_visualizations)))
            montage_path = os.path.join(case_vis_dir, f"volume_montage_{case_id}.png")
            cv2.imwrite(montage_path, volume_montage)
            print(f"Volume montage saved to: {montage_path}")

    # 将结果写入CSV文件
    with open(metrics_csv, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for k, v in metric_dict.items():
            dice_score = 2 * v['tp'] / (2 * v['tp'] + v['fp'] + v['fn'])
            print(model_cfg)
            print(f"label: {k}, dice: {dice_score}")
            # 写入模型配置、标签ID、组件值和Dice分数
            csv_writer.writerow([model_cfg, k, v['tp'], v['fp'], v['fn'], dice_score])

    # 将时间信息写入CSV
    with open(timing_csv, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for case_id, timing_info in timing_dict.items():
            csv_writer.writerow([
                model_cfg,
                case_id,
                timing_info['num_slices'],
                timing_info['total_time'],
                timing_info['avg_time_per_slice'],
                timing_info['speed_fps']
            ])
    # 计算并打印模型级别的时间汇总
    total_slices = sum([t['num_slices'] for t in timing_dict.values()])
    total_time = sum([t['total_time'] for t in timing_dict.values()])
    avg_time = total_time / total_slices if total_slices > 0 else 0
    avg_fps = total_slices / total_time if total_time > 0 else 0

    print(f"\nModel {model_cfg} Summary:")
    print(f"  Total slices processed: {total_slices}")
    print(f"  Total propagation time: {total_time:.2f}s")
    print(f"  Average time per slice: {avg_time * 1000:.1f}ms")
    print(f"  Average speed: {avg_fps:.1f} fps")

print(f"\n{'=' * 50}")
print(f"Inference completed. Results saved to: {visualization_dir}")
print(f"  - Metrics CSV: {metrics_csv}")
print(f"  - Timing CSV: {timing_csv}")
