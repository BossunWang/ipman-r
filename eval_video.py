"""
This script can be used to evaluate a trained model on 3D pose/shape and masks/part segmentation. You first need to download the datasets and preprocess them.
Example usage:
```
python3 eval.py --checkpoint=data/model_checkpoint.pt --dataset=h36m-p1 --log_freq=20
```
Running the above command will compute the MPJPE and Reconstruction Error on the Human3.6M dataset (Protocol I). The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. Human3.6M Protocol 1 ```--dataset=h36m-p1```
2. Human3.6M Protocol 2 ```--dataset=h36m-p2```
3. 3DPW ```--dataset=3dpw```
4. LSP ```--dataset=lsp```
5. MPI-INF-3DHP ```--dataset=mpi-inf-3dhp```
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
# for cluster rendering
import argparse
from tqdm import tqdm
import torchgeometry as tgm
from torchvision.transforms import Normalize


import config
import constants
from models import hmr, SMPL
from datasets import BaseDatasetEval
from utils.imutils import uncrop
from utils.pose_utils import reconstruction_error, compute_accel, compute_error_accel
from utils.geometry import batch_rot2aa, batch_rectify_pose

# from utils.part_utils import PartRenderer
from vis_utils.world_vis import vis_vert_with_ground

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
parser.add_argument('--dataset', default='h36m-p1', help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=32, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')
parser.add_argument('--vis_path', default=None, help='save the world frame visualizations here')
parser.add_argument('--visualize', default=False, action='store_true', help='generate visualization?')
parser.add_argument('--eval_stability', default=False, action='store_true', help='compute stability metrics?')


def cam_to_world(vertices, translation, cam_r, cam_t):
    """
    Convert vertices from camera coordinates to world coordinates.
    """
    # apply predicted translation to the mesh (as tb = -tc)
    vertices = vertices + translation[:, None, :]

    cam_r = cam_r.to(torch.float32)
    cam_t = cam_t.to(torch.float32)

    # cam extrinsics
    mm2m = 1000
    R = cam_r
    t = -torch.bmm(R, cam_t) / mm2m  # t= -RC
    # t = cam_t.to(torch.float32) / mm2m

    # reverse camera to go from camera to world
    R_T = R.permute(0, 2, 1)
    t_w = -torch.bmm(R_T, t)

    # apply extrinsics
    vertices_world = torch.einsum('bij,bkj->bki', R_T, vertices)
    vertices_world = vertices_world + t_w.squeeze()[:, None, :]
    return vertices_world


def run_evaluation(model, dataset_name, dataset, result_file, vis_path,
                   batch_size=32, img_res=224, 
                   num_workers=32, shuffle=False, log_freq=50, visualize=False, eval_stability=False):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Transfer model to the GPU
    model.to(device)

    # Load SMPL model
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)
    smpl_male = SMPL(config.SMPL_MODEL_DIR,
                     gender='male',
                     create_transl=False).to(device)
    smpl_female = SMPL(config.SMPL_MODEL_DIR,
                       gender='female',
                       create_transl=False).to(device)

    # renderer = PartRenderer()

    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()

    save_results = result_file is not None
    # Disable shuffling if you want to save the results
    if save_results:
        shuffle=False
    # Create dataloader for the dataset
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # Pose metrics
    # MPJPE and Reconstruction error for the non-parametric and parametric shapes
    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    mpjpe_smpl = np.zeros(len(dataset))
    recon_err_smpl = np.zeros(len(dataset))
    v2v_err = np.zeros(len(dataset))

    # Acceleration metrics
    accel_ = np.zeros(len(dataset))
    accel_err_ = np.zeros(len(dataset))

    gt_bos_accumulator = []
    gt_contact_accumulator = []
    gt_contact_mask_accumulator = []
    pred_bos_accumulator = []
    pred_contact_accumulator = []
    pred_contact_mask_accumulator = []

    # Shape metrics
    # Mean per-vertex error
    shape_err = np.zeros(len(dataset))
    shape_err_smpl = np.zeros(len(dataset))

    # Mask and part metrics
    # Accuracy
    accuracy = 0.
    parts_accuracy = 0.
    # True positive, false positive and false negative
    tp = np.zeros((2,1))
    fp = np.zeros((2,1))
    fn = np.zeros((2,1))
    parts_tp = np.zeros((7,1))
    parts_fp = np.zeros((7,1))
    parts_fn = np.zeros((7,1))
    # Pixel count accumulators
    pixel_count = 0
    parts_pixel_count = 0

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    eval_pose = False
    eval_masks = False
    eval_parts = False
    # Choose appropriate evaluation for each dataset
    if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2'  \
            or dataset_name == 'h36m-test-s1' or dataset_name == '3dpw' \
            or dataset_name == 'mpi-inf-3dhp' \
            or dataset_name == 'rich-val-onlycam0' or dataset_name == 'rich-test-onlycam0'\
            or dataset_name == 'rich-val' or dataset_name == 'rich-test':
        eval_pose = True
    elif dataset_name == 'lsp':
        eval_masks = True
        eval_parts = True
        annot_path = config.DATASET_FOLDERS['upi-s1h']

    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14

    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    # Iterate over the entire dataset
    for step, video_name in enumerate(tqdm(dataset)):
        video_path = os.path.join(dataset_name, video_name)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            h, w, c = frame.shape
            offset_w = h - w if h > w else 0
            offset_h = w - h if w > h else 0

            frame = cv2.copyMakeBorder(frame, offset_h // 2, offset_h // 2, offset_w // 2, offset_w // 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            rgb_img = cv2.resize(frame, (constants.IMG_RES, constants.IMG_RES))
            cv2.imshow("resize", rgb_img)
            cv2.waitKey(1)
            rgb_img = np.transpose(rgb_img[:, :, ::-1].astype('float32'), (2, 0, 1)) / 255.0
            rgb_img = torch.from_numpy(rgb_img).float()
            img_tensor = normalize_img(rgb_img).unsqueeze(0).to(device)
            curr_batch_size = img_tensor.shape[0]

            with torch.no_grad():
                pred_rotmat, pred_betas, pred_camera = model(img_tensor)
                pred_pose = batch_rot2aa(pred_rotmat.view(-1, 3, 3)).view(curr_batch_size, -1)
                pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
                pred_vertices = pred_output.vertices
                pred_joints = pred_output.joints

                # move to world coordinates - first correct the predicted pose using gt camera rotation, then correct the translation by pelvis alignment

                pred_vertices_world = pred_vertices
                pred_joints_world = pred_joints[:, 25:, :]
                # compute pred transl world by equating it to the gt world pelvis
                pred_pelvis_world = (pred_joints_world[:, 2, :] + pred_joints_world[:, 3, :]) / 2
                pred_transl_world = pred_pelvis_world - pred_pelvis_world

                pred_in_bos_label, pred_contact_metric, pred_contact_mask, _ = vis_vert_with_ground(pred_vertices_world,
                                                                                                 pred_transl_world[:, None, :],
                                                                                                 seq_name=os.path.splitext(video_name)[0],
                                                                                                 vis_path=vis_path,
                                                                                                 start_idx=step * curr_batch_size,
                                                                                                 sub_sample=1,
                                                                                                 imgnames=[f'{os.path.splitext(video_name)[0]}_{frame_count}.png'],
                                                                                                 smpl_batch_size=curr_batch_size,
                                                                                                 ground_offset=0,
                                                                                                 visualize=visualize)
            frame_count += 1
        cap.release()


if __name__ == '__main__':
    args = parser.parse_args()
    model = hmr(config.SMPL_MEAN_PARAMS)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    dataset = sorted(os.listdir(args.dataset))

    # Setup evaluation dataset
    # dataset = BaseDatasetEval(None, args.dataset, is_train=False)
    # Run evaluation
    run_evaluation(model, args.dataset, dataset, args.result_file, args.vis_path,
                   batch_size=args.batch_size,
                   shuffle=args.shuffle,
                   log_freq=args.log_freq,
                   num_workers=args.num_workers,
                   visualize=args.visualize,
                   eval_stability=args.eval_stability)
