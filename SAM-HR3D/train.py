from dataset.datasets import load_data_volume
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
from torch.optim import AdamW
import numpy as np
import logging
from utils.script_util import save_checkpoint
import sys
from monai.losses import DiceCELoss, DiceLoss
from modeling.image_encoder import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
import torch.nn.functional as F
from modeling.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
import torch
from modeling.prompt_encoder import PromptEncoder, TwoWayTransformer
import torch.nn as nn
from functools import partial
import os
from utils.util import setup_logger





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon"])
    parser.add_argument("--snapshot_path", default="", type=str)
    parser.add_argument("--data_prefix", default="", type=str)
    parser.add_argument("--rand_crop_size", default=0, nargs='+', type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--max_epoch", default=500, type=int)
    parser.add_argument("--eval_interval", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument("-tolerance", default=5, type=int)
    parser.add_argument("--print_params", action="store_true", default=True, help="打印模型参数量")

    args = parser.parse_args()
    device = args.device
    if args.rand_crop_size == 0:
        if args.data in ["pancreas", "lits", "colon", "kits"]:
            args.rand_crop_size = (128, 128, 128)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    args.snapshot_path = os.path.join(args.snapshot_path, args.data)
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger("train")
    logger.info(str(args))
    train_data = load_data_volume(
        data=args.data,
        path_prefix=args.data_prefix,
        batch_size=1,
        augmentation=True,
        split="train",
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker=args.num_worker
    )
    val_data = load_data_volume(
        data=args.data,
        path_prefix=args.data_prefix,
        batch_size=1,
        augmentation=False,
        split="val",
        deterministic=True,
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker=args.num_worker
    )
    sam = sam_model_registry["vit_b"](checkpoint="ckpt/sam_vit_b_01ec64.pth")
    mask_generator = SamAutomaticMaskGenerator(sam)
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
    img_encoder.load_state_dict(mask_generator.predictor.model.image_encoder.state_dict(), strict=False)
    del sam
    img_encoder.to(device)

    for p in img_encoder.parameters():
        p.requires_grad = False
    img_encoder.depth_embed.requires_grad = True
    for p in img_encoder.slice_embed.parameters():
        p.requires_grad = True
    for i in img_encoder.blocks:
        for p in i.norm1.parameters():
            p.requires_grad = True
        for p in i.adapter.parameters():
            p.requires_grad = True
        for p in i.norm2.parameters():
            p.requires_grad = True
        for p in i.ipf_projection1.parameters():
            p.requires_grad = True
        for p in i.ipf_triplet_attn.parameters():
            p.requires_grad = True
        for p in i.ipf_projection2.parameters():
            p.requires_grad = True
        i.attn.rel_pos_d = nn.Parameter(0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=True)
    for i in img_encoder.neck_3d:
        for p in i.parameters():
            p.requires_grad = True

    prompt_encoder_list = []
    parameter_list = []
    for i in range(4):
        prompt_encoder = PromptEncoder(transformer=TwoWayTransformer(depth=2,
                                                                     embedding_dim=256,
                                                                     mlp_dim=2048,
                                                                     num_heads=8))
        prompt_encoder.to(device)
        prompt_encoder_list.append(prompt_encoder)
        parameter_list.extend([i for i in prompt_encoder.parameters() if i.requires_grad])

    mask_decoder = VIT_MLAHead(img_size=96, num_classes=2)
    mask_decoder.to(device)



    # 差异化学习率和权重衰减
    encoder_params = [
        {"params": [p for n, p in img_encoder.named_parameters() if "SFP" in n and p.requires_grad], "lr": 5e-5,
         "weight_decay": 2e-2},
        {"params": [p for n, p in img_encoder.named_parameters() if "adapter" in n and p.requires_grad], "lr": 5e-5,
         "weight_decay": 1e-2},
        {"params": [p for n, p in img_encoder.named_parameters() if
                    "ipf" not in n and "adapter" not in n and p.requires_grad], "lr": 1e-4, "weight_decay": 1e-2}
    ]
    encoder_opt = AdamW(encoder_params)
    encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(encoder_opt, T_max=args.max_epoch)
    feature_opt = AdamW(parameter_list, lr=2e-4, weight_decay=1e-2)
    feature_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(feature_opt, T_max=args.max_epoch)
    decoder_opt = AdamW([i for i in mask_decoder.parameters() if i.requires_grad], lr=4e-4, weight_decay=1e-2)
    decoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(decoder_opt, T_max=args.max_epoch)
    dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
    loss_cal = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.7, lambda_ce=0.3)
    best_loss = np.inf
    patch_size = args.rand_crop_size[0]
    for epoch_num in range(args.max_epoch):
        loss_summary = []
        img_encoder.train()
        for module in prompt_encoder_list:
            module.train()
        mask_decoder.train()
        for idx, (img, seg, spacing) in enumerate(train_data):
            logger.info(f"Epoch {epoch_num}, Iter {idx}, seg sum: {seg.sum()}")
            out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
            input_batch = out.to(device)
            input_batch = input_batch[0].transpose(0, 1)
            batch_features, feature_list = img_encoder(input_batch)
            feature_list.append(batch_features)
            l = len(torch.where(seg == 1)[0])
            points_torch = None
            if l > 0:
                sample = np.random.choice(np.arange(l), 10, replace=True)
                x = torch.where(seg == 1)[1][sample].unsqueeze(1)
                y = torch.where(seg == 1)[3][sample].unsqueeze(1)
                z = torch.where(seg == 1)[2][sample].unsqueeze(1)
                points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
                points_torch = points.to(device)
            l = len(torch.where(seg < 10)[0])
            sample = np.random.choice(np.arange(l), 20, replace=True)
            x = torch.where(seg < 10)[1][sample].unsqueeze(1)
            y = torch.where(seg < 10)[3][sample].unsqueeze(1)
            z = torch.where(seg < 10)[2][sample].unsqueeze(1)
            points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
            points_torch_negative = points.to(device)
            if points_torch is not None:
                points_torch = torch.cat([points_torch, points_torch_negative], dim=0)
            else:
                points_torch = points_torch_negative
            logger.info(
                f"Epoch {epoch_num}, Iter {idx}, Positive: {10 if points_torch is not None and l > 0 else 0}, Negative: {points_torch_negative.shape[0]}, Points shape: {points_torch.shape}")
            new_feature = []
            for i, (feature, prompt_encoder) in enumerate(zip(feature_list, prompt_encoder_list)):
                if i == 3:
                    new_feature.append(
                        prompt_encoder(feature, points_torch.clone(), [patch_size, patch_size, patch_size])
                    )
                else:
                    new_feature.append(feature)
            img_resize = F.interpolate(img[:, 0].permute(0, 2, 3, 1).unsqueeze(1).to(device),
                                       scale_factor=64 / patch_size,
                                       mode='trilinear')
            new_feature.append(img_resize)
            masks = mask_decoder(new_feature, 2, patch_size // 64)
            masks = masks.permute(0, 1, 4, 2, 3)
            seg = seg.to(device)
            seg = seg.unsqueeze(1)
    
            l2_loss = 0
            for block in img_encoder.blocks:
                for name, param in block.named_parameters():
                    if "ipf" in name:
                        l2_loss += torch.norm(param, p=2)
            loss = loss_cal(masks, seg) + 1e-4 * l2_loss
            loss_summary.append(loss.detach().cpu().numpy())
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            feature_opt.zero_grad()
            loss.backward()
            logger.info(
                f"epoch: {epoch_num}/{args.max_epoch}, iter: {idx}/{len(train_data)}, loss: {loss_summary[-1].flatten()[0]}")
            torch.nn.utils.clip_grad_norm_(img_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(mask_decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(prompt_encoder_list[-1].parameters(), 1.0)
            encoder_opt.step()
            feature_opt.step()
            decoder_opt.step()
        encoder_scheduler.step()
        feature_scheduler.step()
        decoder_scheduler.step()
        logger.info("- Train metrics: " + str(np.mean(loss_summary)))

        img_encoder.eval()
        for module in prompt_encoder_list:
            module.eval()
        mask_decoder.eval()
        with torch.no_grad():
            loss_summary = []
            for idx, (img, seg, spacing) in enumerate(val_data):
                logger.info(f"Val Epoch {epoch_num}, Iter {idx}, seg sum: {seg.sum()}")
                out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
                input_batch = out.to(device)
                input_batch = input_batch[0].transpose(0, 1)
                batch_features, feature_list = img_encoder(input_batch)
                feature_list.append(batch_features)
                l = len(torch.where(seg == 1)[0])
                points_torch = None
                if l > 0:
                    sample = np.random.choice(np.arange(l), 10, replace=True)
                    x = torch.where(seg == 1)[1][sample].unsqueeze(1)
                    y = torch.where(seg == 1)[3][sample].unsqueeze(1)
                    z = torch.where(seg == 1)[2][sample].unsqueeze(1)
                    points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
                    points_torch = points.to(device)
                l = len(torch.where(seg < 10)[0])
                sample = np.random.choice(np.arange(l), 10, replace=True)
                x = torch.where(seg < 10)[1][sample].unsqueeze(1)
                y = torch.where(seg < 10)[3][sample].unsqueeze(1)
                z = torch.where(seg < 10)[2][sample].unsqueeze(1)
                points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
                points_torch_negative = points.to(device)
                if points_torch is not None:
                    points_torch = torch.cat([points_torch, points_torch_negative], dim=1)
                else:
                    points_torch = points_torch_negative
                logger.info(
                    f"Val Epoch {epoch_num}, Iter {idx}, Positive: {10 if points_torch is not None and l > 0 else 0}, Negative: {points_torch_negative.shape[0]}, Points shape: {points_torch.shape}")
                new_feature = []
                for i, (feature, prompt_encoder) in enumerate(zip(feature_list, prompt_encoder_list)):
                    if i == 3:
                        new_feature.append(
                            prompt_encoder(feature, points_torch.clone(), [patch_size, patch_size, patch_size])
                        )
                    else:
                        new_feature.append(feature)
                img_resize = F.interpolate(img[:, 0].permute(0, 2, 3, 1).unsqueeze(1).to(device),
                                           scale_factor=64 / patch_size,
                                           mode='trilinear')
                new_feature.append(img_resize)
                masks = mask_decoder(new_feature, 2, patch_size // 64)
                masks = masks.permute(0, 1, 4, 2, 3)
                seg = seg.to(device)
                seg = seg.unsqueeze(1)
                loss = dice_loss(masks, seg)
                loss_summary.append(loss.detach().cpu().numpy())
                logger.info(
                    f"epoch: {epoch_num}/{args.max_epoch}, iter: {idx}/{len(val_data)}, loss: {loss_summary[-1].flatten()[0]}")
            logger.info("- Val metrics: " + str(np.mean(loss_summary)))

            is_best = False
            if np.mean(loss_summary) < best_loss:
                best_loss = np.mean(loss_summary)
                is_best = True
            save_checkpoint({"epoch": epoch_num + 1,
                             "best_val_loss": best_loss,
                             "encoder_dict": img_encoder.state_dict(),
                             "decoder_dict": mask_decoder.state_dict(),
                             "feature_dict": [i.state_dict() for i in prompt_encoder_list],
                             "encoder_opt": encoder_opt.state_dict(),
                             "feature_opt": feature_opt.state_dict(),
                             "decoder_opt": decoder_opt.state_dict()
                             },
                            is_best=is_best,
                            checkpoint=args.snapshot_path)
            logger.info("- Val metrics best: " + str(best_loss))


if __name__ == "__main__":
    main()
