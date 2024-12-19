import copy

import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
# import open_clip as clip
import torch
import clip
from PIL import Image
from pathlib import Path
import random, glob, os
from tqdm import tqdm
import configargparse
import imageio
from funcs import vis_seg, DistinctColors
import torchvision
from featup.util import norm, unnorm
from featup.plotting import plot_feats, plot_lang_heatmaps
import torchvision.transforms as T
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CLIPNetwork:
    def __init__(self, device):
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.clip_model_type = "ViT-B-16"
        self.clip_model_pretrained = 'laion2b_s34b_b88k'
        self.clip_n_dims = 512
        model, _ = clip.load("ViT-B/16", device='cuda')
        model.eval()
        model = model.float()
        self.tokenizer = clip.tokenize
        self.model = model.to(device)

        self.negatives = ("object", "things", "stuff", "texture")
        # self.negatives = ("object", "things", "stuff", "texture")
        # self.negatives = ("object")
        self.positives = (" ",)
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to(device)
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to(device)
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        # embed: 32768x512
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)
        output = torch.mm(embed, p.T)
        # print(output.shape)
        positive_vals = output[..., positive_id: positive_id + 1]
        negative_vals = output[..., len(self.positives) :]
        # negative_vals = torch.cat([output[..., :positive_id], output[..., positive_id + 1: len(self.positives)]],
        #                           dim=-1)
        # print(negative_vals.shape)
        repeated_pos = positive_vals.repeat(1, negative_vals.shape[1])

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)
        softmax = torch.softmax(10 * sims, dim=-1)
        best_id = softmax[..., 0].argmin(dim=1)
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
               :, 0, :
               ]

    def encode_image(self, input, mask=None):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input, mask=mask)

    def encode_text(self, text_list, device):
        text = self.tokenizer(text_list).to(device)
        return self.model.encode_text(text)

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.positives]
            ).to(self.neg_embeds.device)
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def set_semantics(self, text_list):
        self.semantic_labels = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.semantic_labels]).to("cuda")
            self.semantic_embeds = self.model.encode_text(tok_phrases)
        self.semantic_embeds /= self.semantic_embeds.norm(dim=-1, keepdim=True)

    def get_semantic_map(self, sem_map: torch.Tensor) -> torch.Tensor:
        # embed: 3xhxwx512
        n_levels, h, w, c = sem_map.shape
        pos_num = self.semantic_embeds.shape[0]
        phrases_embeds = torch.cat([self.semantic_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(sem_map.dtype)
        sem_pred = torch.zeros(n_levels, h, w)
        for i in range(n_levels):
            output = torch.mm(sem_map[i].view(-1, c), p.T)
            softmax = torch.softmax(10 * output, dim=-1)
            sem_pred[i] = torch.argmax(softmax, dim=-1).view(h, w)
            sem_pred[i][sem_pred[i] >= pos_num] = -1
        return sem_pred.long()

    def get_max_across(self, sem_map):
        n_phrases = len(self.positives)
        n_phrases_sims = [None for _ in range(n_phrases)]

        n_levels, h, w, _ = sem_map.shape
        clip_output = sem_map.permute(1, 2, 0, 3).flatten(0, 1)

        n_levels_sims = [None for _ in range(n_levels)]
        for i in range(n_levels):
            for j in range(n_phrases):
                probs = self.get_relevancy(clip_output[..., i, :], j)
                pos_prob = probs[..., 0:1]
                n_phrases_sims[j] = pos_prob
            n_levels_sims[i] = torch.stack(n_phrases_sims)

        relev_map = torch.stack(n_levels_sims).view(n_levels, n_phrases, h, w)
        return relev_map

def pad_img(img):
    h, w, _ = img.shape
    l = max(w, h)
    pad = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        pad[:, (h - w) // 2:(h - w) // 2 + w, :] = img
    else:
        pad[(w - h) // 2:(w - h) // 2 + h, :, :] = img
    return pad


def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep


def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.

    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]

    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr

    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx


def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred = torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

parser = configargparse.ArgumentParser()

parser.add_argument("--scene_name", type=str, default='sofa')


args = parser.parse_args()
images_path = f"/data/3D-OVS/Scenes/{args.scene_name}/images/"
seg_path = f"/data/3D-OVS/Scenes/{args.scene_name}/segmentations/classes.txt"
save_path = f'/data/3D-OVS/sam_mask/{args.scene_name}'
# save_path = f'/data/3D-OVS/mask_clip_feature/{args.scene_name}'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam = sam_model_registry["vit_h"](checkpoint='/home/zzz/wyj/LangSplat/ckpts/sam_vit_h_4b8939.pth').to('cuda')
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.7,
    box_nms_thresh=0.7,
    stability_score_thresh=0.85,
    crop_n_layers=0,
    crop_n_points_downscale_factor=1,
    min_mask_region_area=100,
)
predictor = SamPredictor(sam)


image_paths = sorted(glob.glob(f'{images_path}/*'))
feature_paths = sorted(glob.glob(f'/data/3D-OVS/clip_features/{args.scene_name}/*'))

os.makedirs(save_path, exist_ok=True)


clip_network = CLIPNetwork(device)

    # read class names
with open(seg_path, 'r') as f:
    lines = f.readlines()
    classes = [line.strip() for line in lines]
    classes.sort()
clip_network.set_positives(classes)
input_size = 224
transform = T.Compose([
    T.Resize((input_size, input_size)),
    T.ToTensor(),
    norm
])
upsampler = torch.hub.load("mhamilton723/FeatUp", 'maskclip', use_norm=False).to(device)

from featup.featurizers.maskclip.clip import tokenize
# #


text = tokenize(classes).to(device)
text_features = upsampler.model.model.encode_text(text).squeeze().to(torch.float32)
text_features = F.normalize(text_features, dim=1)
for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
    name = image_path.split('.')[0].split('/')[-1]
    down_sample = 8
    image_path = Path(image_path)
    image = Image.open(image_path)
    img_hw = (image.height//down_sample, image.width//down_sample)
    image = image.resize((image.width//down_sample, image.height//down_sample))
    image = np.array(image)
    predictor.set_image(image)
    clip_feat = upsampler(transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device))

    feature_shape_wo_dim = (image.shape[0], image.shape[1])
    clip_feat = F.interpolate(clip_feat, size=feature_shape_wo_dim).squeeze(0)

    feature_map = rearrange(clip_feat.squeeze(0), 'c h w -> (h w) c').cuda()
    feature_map_normalized = F.normalize(feature_map, dim=1).detach()  # [N1,D]

    relevancy_map = torch.mm(feature_map_normalized, text_features.T).detach()  # [N1,N2=num_classes]

    p_class = rearrange(relevancy_map, '(h w ) c -> 1 c h w', h=feature_shape_wo_dim[0],
                        w=feature_shape_wo_dim[1])  # [N1,N2]
    # p_class = relevancy_map  # [N1,N2]
    class_index = torch.argmax(p_class, dim=1)

    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)

    masks_default, masks_s, masks_m, masks_l = \
        masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)

    masks = {'s':masks_s,
             'l':masks_l,
             'm':masks_m,
             }
    # print(masks_s)
    seg = []
    for mask in masks_m:
        seg.append(torch.tensor(mask['segmentation']))
    for mask in masks_l:
        seg.append(torch.tensor(mask['segmentation']))
    for mask in masks_s:
        seg.append(torch.tensor(mask['segmentation']))
    seg = torch.stack(seg, dim=0)

    mask_clip_feature = []
    for mask in seg:
        clip_feature = clip_feat * mask[None, ...].to(device)
        mask_clip_feature.append((rearrange(clip_feature, 'c h w -> c (h w)') / torch.sum(mask)).sum(1).cpu())


    mask_clip_feature = torch.stack(mask_clip_feature).cpu()


    mask_clip_feature = F.normalize(mask_clip_feature, dim=1)

    masks['all_mask'] = seg
    masks['clip_feature'] = mask_clip_feature

    torch.save(masks, os.path.join(save_path, name + '.pt'))

