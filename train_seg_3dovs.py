
from opt import config_parser

from models.losses import MaskCriterion
from models.matcher import HungarianMatcher
from renderer import *
from funcs import *
from torch.utils.tensorboard import SummaryWriter
import datetime
import torchvision
from dataLoader import dataset_dict
import sys
from pathlib import Path

from einops import repeat, rearrange, reduce

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from featup.featurizers.maskclip.clip import tokenize
renderer = OctreeRender_trilinear_fast


class OpenCLIPNetwork:
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
        model = torch.hub.load("mhamilton723/FeatUp", 'maskclip', use_norm=False).to(device)
        model.eval()

        self.tokenizer = tokenize
        self.model = model.to(device)

        self.negatives = ("object", "things", "stuff", "texture")
        self.positives = (" ",)
        with torch.no_grad():
            # tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to(device)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to(device)
            self.pos_embeds = model.model.model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to(device)
            self.neg_embeds = model.model.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        # embed: 32768x512
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)
        output = torch.mm(embed, p.T)
        positive_vals = output[..., positive_id: positive_id + 1]
        negative_vals = output[..., len(self.positives):]
        repeated_pos = positive_vals.repeat(1, len(self.negatives))

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
        return self.model.model.model.encode_text(text)

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = self.tokenizer(text_list).cuda()
            self.pos_embeds = self.model.model.model.encode_text(tok_phrases).squeeze().to(torch.float32)
        self.pos_embeds = F.normalize(self.pos_embeds, dim=1)

    def set_semantics(self, text_list):
        self.semantic_labels = text_list
        with torch.no_grad():
            tok_phrases = self.tokenizer(text_list).cuda()
            self.semantic_embeds = self.model.model.model.encode_text(tok_phrases).squeeze().to(torch.float32)
        self.semantic_embeds = F.normalize(self.semantic_embeds, dim=-1)
        # print(self.semantic_embeds)
    def get_semantic_map(self, sem_map: torch.Tensor) -> torch.Tensor:
        phrases_embeds = self.semantic_embeds
        p = phrases_embeds.to(sem_map.dtype)

        output = sem_map @ p.T

        return output

    def get_max_across(self, sem_map):
        n_phrases = len(self.positives)
        n_phrases_sims = [None for _ in range(n_phrases)]

        n_levels, _ = sem_map.shape
        clip_output = F.normalize(sem_map, dim=-1)

        # n_levels_sims = [None for _ in range(n_levels)]
        # for i in range(n_levels):
        for j in range(n_phrases):
            probs = self.get_relevancy(clip_output, j)
            pos_prob = probs[..., 0:1]
            n_phrases_sims[j] = pos_prob
        # n_levels_sims[i] = torch.stack(n_phrases_sims)

        relev_map = torch.stack(n_phrases_sims)
        # print(relev_map.shape)
        return relev_map

def smooth(mask):
    h, w = mask.shape[:2]
    im_smooth = mask.copy()
    scale = 3
    for i in range(h):
        for j in range(w):
            square = mask[max(0, i-scale) : min(i+scale+1, h-1),
                          max(0, j-scale) : min(j+scale+1, w-1)]
            im_smooth[i, j] = np.argmax(np.bincount(square.reshape(-1)))
    return im_smooth

def InfiniteSampler(n):
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr + self.batch]


@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='train', downsample=args.downsample_test, is_stack=True)
    print(args.downsample_test)
    test_dataset.read_classes_names()
    if args.has_segmentation_maps:
        test_dataset.read_segmentation_maps()
    c2ws = test_dataset.render_path
    classes = test_dataset.classes
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.change_to_feature_mode(device)
    tensorf.load(ckpt)
    tensorf.eval()

    logfolder = os.path.dirname(args.ckpt)



def reconstruction(args):
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    if args.dataset_name == 'replica':
        scene_name = args.datadir.split('/')[-2]
    else:
        scene_name = args.datadir.split('/')[-1]
    feature_train_dataset = dataset(args.datadir, split='train', patch_size=args.patch_size,
                                    downsample=args.ray_downsample_train, is_stack=True, clip_input=args.clip_input)
    feature_train_dataset.read_classes_names()
    if args.has_segmentation_maps:
        feature_train_dataset.read_segmentation_maps()
    feature_train_dataset.read_sam_mask(scene_name)
    patch_train_dataset = dataset(args.datadir, split='train', patch_size=args.patch_size,
                                  downsample=args.patch_downsample_train, is_stack=True)



    patch_train_dataset.read_sam_mask(scene_name)

    ndc_ray = args.ndc_ray

    classes = feature_train_dataset.classes
    matcher = HungarianMatcher()
    mask_loss = MaskCriterion(matcher, losses=["features", "masks"]).cuda()

    # init log file
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'

    os.makedirs(logfolder, exist_ok=True)
    init_logger(Path(logfolder))
    logger.info(args)
    logger.info(f'classes: {classes}')

    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)

    patch_train_loader = iter(torch.utils.data.DataLoader(patch_train_dataset, batch_size=args.patch_num,
                                                          sampler=InfiniteSamplerWrapper(patch_train_dataset),
                                                          num_workers=0, pin_memory=True))

    # init parameters
    aabb = feature_train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))

    # load pre-trained nerf
    assert args.ckpt is not None, 'Have to be pre-trained to get the density field!'

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    tensorf.init_mask_query(num_class=len(classes), N_query=args.num_query)
    tensorf.change_to_feature_mode(device)


    clip_model = OpenCLIPNetwork(device)
    clip_model.set_positives(classes)
    clip_model.set_semantics(classes)

    grad_vars = tensorf.get_optparam_groups_feature_mod(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    dc = DistinctColors()
    k = 4
    # training loop
    torch.cuda.empty_cache()

    # 均匀选择索引

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout, bar_format='{l_bar}{r_bar}')
    for iteration in pbar:

        batch = next(patch_train_loader)

        rays_train, rgbs_train = batch['rays'], batch['rgbs'].to(
            device)  # [B, H//8, W//8, 6], [B, H, W, 3]

        # if iteration in [3000]:
        #     k = k // 2
        avg_pool = torch.nn.AvgPool2d(k, ceil_mode=True)
        rays_train = avg_pool(rays_train.permute(0,3,1,2)).squeeze(0).permute(1,2,0)
        rays_train = rays_train.squeeze(0)
        # print(rays_train.shape)

        rays_train_shape = [rays_train.shape[0], rays_train.shape[1]]

        rays_train = rays_train.reshape(-1, 6)

        _, mask_feature_map, _, rgb_map = renderer(rays_train, tensorf, chunk=args.chunk_size, N_samples=nSamples,
                                             ndc_ray=ndc_ray, is_train=True, render_feature=True, out_mask=True, device=device)

        mask_query = tensorf.get_mask_query()
        mask_logit = mask_query.cuda() @ mask_feature_map.cuda().T
        mask_logit = mask_logit.view(1, args.num_query, *rays_train_shape)

        sam_mask = batch['masks'].cuda()

        mask_clip = tensorf.get_mask_clip()
        sam_mask_clip = batch['clip']
        pred_mask = {
                     'pred_masks': mask_logit.cuda(),
                     'pred_clip': mask_clip.unsqueeze(0).cuda(),
                     }
        gt_mask = [{"masks": sam_mask.squeeze(0).cuda(),
                    'clip': sam_mask_clip.squeeze(0).cuda(),
                    }]
        mask_l = mask_loss(pred_mask, gt_mask)

        loss = mask_l['loss_mask'] + 0.05 * mask_l['loss_dice'] + mask_l['loss_clip'] + mask_l['loss_extra_mask']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        #############
        ###logging###
        #############
        if iteration % args.progress_refresh_rate == 0:
            feature_loss = torch.tensor(0)
            if iteration >= 1:
                # Print the current values of the losses.
                pbar.set_description(
                    f"loss={loss.detach().item():.2f} | mask_clip_loss={mask_l['loss_clip'].detach().item():.3f} | mask_loss={mask_l['loss_mask'].detach().item() + 0.05 * mask_l['loss_dice'].detach().item():.2f}"
                )
            else:

                pbar.set_description(
                    f"loss={loss.detach().item():.2f} | feature_loss={feature_loss.detach().item():.2f}"
                )

        if iteration + 1 == (args.n_iters - 1) or ((iteration) % (args.progress_refresh_rate * 100) == 0):
            # if iteration == 1:
            with (torch.no_grad()):
                savePath = f'{logfolder}/imgs_vis'
                os.makedirs(savePath, exist_ok=True)
                IoUs, accuracies = [], []
                for i, frame_idx in tqdm(enumerate(feature_train_dataset.idxes)):
                    gt_seg = feature_train_dataset.seg_maps[i]  # [H*W=N1, n_classes]
                    W, H = feature_train_dataset.img_wh
                    rays = feature_train_dataset.all_rays_stack[frame_idx].reshape(-1, 6)
                    rgbs = feature_train_dataset.all_rgbs_stack[frame_idx].reshape(-1, 3)

                    feature_map, mask_feature_map, _, _ = renderer(rays, tensorf, chunk=1024, N_samples=nSamples,
                                                                   ndc_ray=ndc_ray, is_train=False, render_feature=True,
                                                                   out_mask=True, device=device)
                    colormap_saving(rearrange(mask_feature_map, '(h w) c -> h w c', h=H, w=W), colormap_options, f'{savePath}/{iteration:05d}_{frame_idx:02d}_maskfeature.png')

                    mask_query = tensorf.get_mask_query()
                    mask_logit = mask_query.cuda() @ mask_feature_map.cuda().T

                    mask_map = F.sigmoid(mask_logit.view(1, args.num_query, W, H)).view(args.num_query, -1)
                    mask_clip_feature = tensorf.get_mask_clip()

                    mask_clip_feature = F.normalize(mask_clip_feature, dim=-1)


                    valid_map = clip_model.get_semantic_map(mask_clip_feature.cuda())
                    valid_map_min = torch.min(valid_map, dim=0, keepdim=True).values
                    valid_map_max = torch.max(valid_map, dim=0, keepdim=True).values
                    valid_map = (valid_map - valid_map_min) / (valid_map_max - valid_map_min)
                    valid_map = torch.softmax(10 * valid_map, dim=-1).unsqueeze(1)
                    mask_map = fielter(rearrange(mask_map, 'n (h w) -> n h w', h=H, w=W)).view(*mask_map.shape).cuda()


                    masked_class_p = mask_map.unsqueeze(2).repeat(1, 1, len(classes)) * valid_map.repeat(1, W * H, 1)
                    masked_class_p = torch.sum(masked_class_p.cuda(), dim=0)

                    class_index = torch.argmax(masked_class_p, dim=1)
                    class_index = smooth(class_index.view(H, W).cpu().numpy())
                    class_index = torch.from_numpy(class_index).view(-1)

                    segmentation_map = vis_seg(dc, class_index, H, W, rgb=rgbs)

                    gt_class_index = torch.argmax(torch.from_numpy(gt_seg), dim=1)

                    gt_segmentation_map = vis_seg(dc, gt_class_index.long(), H, W, rgb=rgbs)
                    if savePath is not None:
                        imageio.imwrite(f'{savePath}/{iteration:05d}_{frame_idx:02d}.png', segmentation_map)
                        imageio.imwrite(f'{savePath}/{iteration:05d}_{frame_idx:02d}_gt.png', gt_segmentation_map)

                    one_hot = F.one_hot(class_index.long(), num_classes=gt_seg.shape[-1])  # [N1, n_classes]
                    one_hot = one_hot.detach().cpu().numpy().astype(np.int8)

                    IoUs.append(jaccard_score(gt_seg, one_hot, average=None))
                    accuracies.append(accuracy_score(gt_seg, one_hot))

                    # write IoUs to log file
                logger.info(f'\n\niteration: {iteration}')
                logger.info(f'overall: mIoU={np.mean(IoUs)}, accuracy={np.mean(accuracies)}\n')

                for i, iou in enumerate(IoUs):
                    logger.info(f'test image {i}: mIoU={np.mean(iou)}, accuracy={accuracies[i]}')
                    logger.info(f'classes iou: {iou}')

    tensorf.save(f'{logfolder}/{args.expname}.th')


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20230417)
    np.random.seed(20230417)

    args = config_parser()

    reconstruction(args)
