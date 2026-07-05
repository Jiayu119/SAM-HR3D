from dataset.datasets import load_data_volume
import argparse
import numpy as np
import logging
from monai.losses import DiceCELoss, DiceLoss
from modeling.image_encoder import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
import torch.nn.functional as F
from modeling.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
import torch
from modeling.prompt_encoder import PromptEncoder, TwoWayTransformer
from functools import partial
import os
from utils.util import setup_logger
import surface_distance
from surface_distance import metrics
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon"])
    parser.add_argument("--snapshot_path", default="", type=str)
    parser.add_argument("--data_prefix", default="", type=str)
    parser.add_argument("--rand_crop_size", default=0, nargs='+', type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num_prompts", default=3, type=int)
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument("--checkpoint", default="last", type=str)
    parser.add_argument("-tolerance", default=5, type=int)
    # 新增：是否进行可视化（可通过命令行控制）
    parser.add_argument("--visualize", action="store_true", help="whether to generate visualization")
    parser.add_argument("--vis_interval", default=5, type=int, help="visualize every N cases")

    args = parser.parse_args()

    if args.checkpoint == "last":
        file = "last.pth.tar"
    else:
        file = "best.pth.tar"

    device = args.device

    if args.rand_crop_size == 0:
        if args.data in ["colon", "pancreas", "lits", "kits"]:
            args.rand_crop_size = (128, 128, 128)

    if isinstance(args.rand_crop_size, int):
        args.rand_crop_size = (args.rand_crop_size,) * 3
    elif len(args.rand_crop_size) == 1:
        args.rand_crop_size = tuple(args.rand_crop_size * 3)
    else:
        args.rand_crop_size = tuple(args.rand_crop_size)

    args.snapshot_path = os.path.join(args.snapshot_path, args.data)

    # 创建可视化保存目录
    vis_dir = os.path.join(args.snapshot_path, "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    setup_logger(logger_name="test", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"test")
    logger.info(str(args))

    test_data = load_data_volume(
        data=args.data,
        batch_size=1,
        path_prefix=args.data_prefix,
        augmentation=False,
        split="test",
        rand_crop_spatial_size=args.rand_crop_size,
        convert_to_sam=False,
        do_test_crop=False,
        deterministic=True,
        num_worker=0
    )

    img_encoder = ImageEncoderViT_3d(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        cubic_window_size=8,
        out_chans=256,
        num_slice=16
    )
    img_encoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["encoder_dict"],
                                strict=True)
    img_encoder.to(device)

    prompt_encoder_list = []
    for i in range(4):
        prompt_encoder = PromptEncoder(transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8
        ))
        prompt_encoder.load_state_dict(
            torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["feature_dict"][i], strict=True)
        prompt_encoder.to(device)
        prompt_encoder_list.append(prompt_encoder)

    mask_decoder = VIT_MLAHead(img_size=96).to(device)
    mask_decoder.load_state_dict(
        torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["decoder_dict"],
        strict=True
    )

    dice_loss = DiceLoss(include_background=False, softmax=False, to_onehot_y=True, reduction="none")

    img_encoder.eval()
    for pe in prompt_encoder_list:
        pe.eval()
    mask_decoder.eval()

    patch_size = args.rand_crop_size[0]

    def model_predict(img, prompt, img_encoder, prompt_encoder_list, mask_decoder):
        out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
        input_batch = out[0].transpose(0, 1)
        batch_features, feature_list = img_encoder(input_batch)
        feature_list.append(batch_features)

        points_torch = prompt.transpose(0, 1)
        new_feature = []
        for i, (feature, pe) in enumerate(zip(feature_list, prompt_encoder_list)):
            if i == 3:
                new_feature.append(
                    pe(feature.to(device), points_torch.clone(), [patch_size, patch_size, patch_size])
                )
            else:
                new_feature.append(feature.to(device))

        img_resize = F.interpolate(
            img[0, 0].permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(device),
            scale_factor=64 / patch_size,
            mode="trilinear"
        )
        new_feature.append(img_resize)

        masks = mask_decoder(new_feature, 2, patch_size // 64)
        masks = masks.permute(0, 1, 4, 2, 3)
        return masks

    with torch.no_grad():
        loss_summary = []
        loss_nsd = []

        for idx, (img, seg, spacing) in enumerate(test_data):
            seg = seg.float()
            prompt = F.interpolate(seg[None, :, :, :, :], img.shape[2:], mode="nearest")[0]
            seg = seg.to(device).unsqueeze(0)
            img = img.to(device)

            seg_pred = torch.zeros_like(prompt).to(device)
            l = len(torch.where(prompt == 1)[0])

            if l == 0:
                logger.warning(f"Case {idx} has no positive label, skipping prompt sampling.")
                continue

            sample = np.random.choice(np.arange(l), args.num_prompts, replace=True)
            x = torch.where(prompt == 1)[1][sample].unsqueeze(1)
            y = torch.where(prompt == 1)[3][sample].unsqueeze(1)
            z = torch.where(prompt == 1)[2][sample].unsqueeze(1)

            x_m = (torch.max(x) + torch.min(x)) // 2
            y_m = (torch.max(y) + torch.min(y)) // 2
            z_m = (torch.max(z) + torch.min(z)) // 2

            d_min = x_m - patch_size // 2
            d_max = x_m + patch_size // 2
            h_min = z_m - patch_size // 2
            h_max = z_m + patch_size // 2
            w_min = y_m - patch_size // 2
            w_max = y_m + patch_size // 2

            d_l = max(0, -d_min)
            d_r = max(0, d_max - prompt.shape[1])
            h_l = max(0, -h_min)
            h_r = max(0, h_max - prompt.shape[2])
            w_l = max(0, -w_min)
            w_r = max(0, w_max - prompt.shape[3])

            points = torch.cat([x - d_min, y - w_min, z - h_min], dim=1).unsqueeze(1).float()
            points_torch = points.to(device)

            d_min = max(0, d_min)
            h_min = max(0, h_min)
            w_min = max(0, w_min)

            img_patch = img[:, :, d_min:d_max, h_min:h_max, w_min:w_max].clone()
            img_patch = F.pad(img_patch, (w_l, w_r, h_l, h_r, d_l, d_r))

            pred = model_predict(img_patch, points_torch, img_encoder, prompt_encoder_list, mask_decoder)
            pred = pred[:, :, d_l:patch_size - d_r, h_l:patch_size - h_r, w_l:patch_size - w_r]
            pred = F.softmax(pred, dim=1)[:, 1]

            seg_pred[:, d_min:d_max, h_min:h_max, w_min:w_max] += pred

            final_pred = F.interpolate(seg_pred.unsqueeze(1), size=seg.shape[2:], mode="trilinear")
            masks = final_pred > 0.5

            loss = 1 - dice_loss(masks, seg)
            loss_summary.append(loss.detach().cpu().numpy())

            ssd = surface_distance.compute_surface_distances(
                (seg == 1)[0, 0].cpu().numpy(),
                (masks == 1)[0, 0].cpu().numpy(),
                spacing_mm=spacing[0].numpy()
            )
            nsd = metrics.compute_surface_dice_at_tolerance(ssd, args.tolerance)
            loss_nsd.append(nsd)

            logger.info(
                f"Case {test_data.dataset.img_dict[idx]} - Dice {loss.item():.6f} | NSD {nsd:.6f}"
            )

            # ====================== 可视化部分（ground truth和预测结果分别展示） ======================
            if args.visualize:  # 改为对每一个 case 进行可视化（移除 % vis_interval）
                try:
                    img_np = img[0, 0].cpu().numpy()
                    gt_np = seg[0, 0].cpu().numpy()
                    pred_np = masks[0, 0].cpu().numpy().astype(np.float32)

                    # 检查并记录形状（debug 用）
                    logger.info(f"Case {idx} shapes → img: {img_np.shape}, gt: {gt_np.shape}, pred: {pred_np.shape}")

                    # 如果形状不匹配，interpolate img 到 gt 的形状（确保一致）
                    if img_np.shape != gt_np.shape:
                        img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float()  # [1,1,D,H,W]
                        img_tensor = F.interpolate(img_tensor, size=gt_np.shape, mode='trilinear', align_corners=False)
                        img_np = img_tensor.squeeze().numpy()
                        logger.info(f"Interpolated img to match gt shape: {img_np.shape}")

                    # 优化切片：使用 gt 中病灶（>0）的 bounding box 中心，确保切到病灶
                    if gt_np.sum() == 0:
                        logger.warning(f"Case {idx} has no positive ground truth, skipping visualization.")
                        raise ValueError("No positive labels in GT")

                    positions = np.where(gt_np > 0)
                    mid_d = (np.min(positions[0]) + np.max(positions[0])) // 2
                    mid_h = (np.min(positions[1]) + np.max(positions[1])) // 2
                    mid_w = (np.min(positions[2]) + np.max(positions[2])) // 2

                    # 确保索引在界内（robust check）
                    mid_d = np.clip(mid_d, 0, gt_np.shape[0] - 1)
                    mid_h = np.clip(mid_h, 0, gt_np.shape[1] - 1)
                    mid_w = np.clip(mid_w, 0, gt_np.shape[2] - 1)

                    # 创建2行3列的子图
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    case_name = test_data.dataset.img_dict[idx]
                    fig.suptitle(f"Case: {case_name} | Dice: {loss.item():.4f} | NSD@{args.tolerance}mm: {nsd:.4f}",
                                 fontsize=16, y=0.98)

                    # 第一行：Ground Truth（只显示原始图像，不显示轮廓）
                    # 轴向
                    axes[0, 0].imshow(img_np[mid_d], cmap='gray')
                    axes[0, 0].set_title('Ground Truth - Axial', fontsize=12, fontweight='bold')
                    axes[0, 0].set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
                    axes[0, 0].text(0.02, 0.98, f'Slice: {mid_d}', transform=axes[0, 0].transAxes,
                                    color='white', fontsize=10, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                    axes[0, 0].axis('off')

                    # 冠状
                    axes[0, 1].imshow(img_np[:, mid_h, :], cmap='gray')
                    axes[0, 1].set_title('Ground Truth - Coronal', fontsize=12, fontweight='bold')
                    axes[0, 1].text(0.02, 0.98, f'Slice: {mid_h}', transform=axes[0, 1].transAxes,
                                    color='white', fontsize=10, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                    axes[0, 1].axis('off')

                    # 矢状
                    axes[0, 2].imshow(img_np[:, :, mid_w], cmap='gray')
                    axes[0, 2].set_title('Ground Truth - Sagittal', fontsize=12, fontweight='bold')
                    axes[0, 2].text(0.02, 0.98, f'Slice: {mid_w}', transform=axes[0, 2].transAxes,
                                    color='white', fontsize=10, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                    axes[0, 2].axis('off')

                    # 第二行：Prediction（显示预测结果轮廓）
                    # 轴向
                    axes[1, 0].imshow(img_np[mid_d], cmap='gray')
                    axes[1, 0].contour(pred_np[mid_d], colors='red', linewidths=2.0, alpha=0.8)
                    axes[1, 0].set_title('Prediction - Axial', fontsize=12, fontweight='bold')
                    axes[1, 0].set_ylabel('Prediction', fontsize=12, fontweight='bold')
                    axes[1, 0].text(0.02, 0.98, f'Slice: {mid_d}', transform=axes[1, 0].transAxes,
                                    color='white', fontsize=10, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                    axes[1, 0].axis('off')

                    # 冠状
                    axes[1, 1].imshow(img_np[:, mid_h, :], cmap='gray')
                    axes[1, 1].contour(pred_np[:, mid_h, :], colors='red', linewidths=2.0, alpha=0.8)
                    axes[1, 1].set_title('Prediction - Coronal', fontsize=12, fontweight='bold')
                    axes[1, 1].text(0.02, 0.98, f'Slice: {mid_h}', transform=axes[1, 1].transAxes,
                                    color='white', fontsize=10, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                    axes[1, 1].axis('off')

                    # 矢状
                    axes[1, 2].imshow(img_np[:, :, mid_w], cmap='gray')
                    axes[1, 2].contour(pred_np[:, :, mid_w], colors='red', linewidths=2.0, alpha=0.8)
                    axes[1, 2].set_title('Prediction - Sagittal', fontsize=12, fontweight='bold')
                    axes[1, 2].text(0.02, 0.98, f'Slice: {mid_w}', transform=axes[1, 2].transAxes,
                                    color='white', fontsize=10, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                    axes[1, 2].axis('off')

                    plt.tight_layout()

                    # 文件名带 dice 和 nsd
                    filename = f"case_{idx:03d}_dice_{loss.item():.4f}_nsd_{nsd:.4f}.png"
                    save_path = os.path.join(vis_dir, filename)
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)

                    logger.info(f"Visualization saved: {filename}")

                except Exception as e:
                    logger.warning(f"Visualization failed for case {idx}: {str(e)}")

        if loss_summary:
            mean_dice = np.mean(loss_summary)
            mean_nsd = np.mean(loss_nsd)
            logger.info(f"Test metrics - Mean Dice: {mean_dice:.6f}")
            logger.info(f"Test metrics - Mean NSD@{args.tolerance}mm: {mean_nsd:.6f}")
        else:
            logger.warning("No valid cases were processed.")


if __name__ == "__main__":
    main()