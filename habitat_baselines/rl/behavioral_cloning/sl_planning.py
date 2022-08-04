import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from skimage.draw import disk
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

BASE_PTH = "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl"
MODEL_DIR = os.path.join(BASE_PTH, "sl_weights/planning/100k_student_1157")
os.makedirs(MODEL_DIR, exist_ok=True)
TB_DIR = os.path.join(BASE_PTH, f"sl_tbs/sl_tb_{time.time()}")
IMG_DIR = os.path.join(BASE_PTH, "sl_planning_1157_imgs")

MAP_PTH = os.path.join(BASE_PTH, "context_maps_100k_student.npy")
GOAL_PTH = os.path.join(BASE_PTH, "context_goals_100k_student.npy")
WPT_MAP_PTH = os.path.join(BASE_PTH, "context_waypoint_maps_100k_student.npy")
WPT_PTH = os.path.join(BASE_PTH, "context_waypoints_100k_student.npy")

# EVAL_MAP_PTH = os.path.join(BASE_PTH, "eval_context_maps_10k_rot.npy")
# EVAL_GOAL_PTH = os.path.join(BASE_PTH, "eval_context_goals_10k_rot.npy")
# EVAL_WPT_MAP_PTH = os.path.join(
#     BASE_PTH, "eval_context_waypoint_maps_10k_rot.npy"
# )
# EVAL_WPT_PTH = os.path.join(BASE_PTH, "eval_context_waypoints_10k_rot.npy")

EVAL_MAP_PTH = os.path.join(BASE_PTH, "data/1157_data/context_maps_1157.npy")
EVAL_GOAL_PTH = os.path.join(BASE_PTH, "data/1157_data/context_goals_1157.npy")
EVAL_WPT_MAP_PTH = os.path.join(
    BASE_PTH, "data/1157_data/context_waypoint_maps_1157.npy"
)
EVAL_WPT_PTH = os.path.join(
    BASE_PTH, "data/1157_data/context_waypoints_1157.npy"
)

TEST_MAP_PTH = os.path.join(BASE_PTH, "test_context_maps.npy")
TEST_GOAL_PTH = os.path.join(BASE_PTH, "test_context_goals.npy")
TEST_WPT_MAP_PTH = os.path.join(BASE_PTH, "test_context_waypoint_maps.npy")
TEST_WPT_PTH = os.path.join(BASE_PTH, "test_context_waypoints.npy")


class Encoder(nn.Module):
    def __init__(self, input_shape, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        in_channels = 2
        cnn_out_dim = int((input_shape[0] // 16) * (input_shape[1] // 16))

        self.encoder = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * cnn_out_dim, 512),
        )
        self.mlp = nn.Sequential(
            nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 3)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        pred_out = self.encoder(x)
        pred_out = self.mlp(pred_out)
        return pred_out


class Planner:
    def __init__(self, mode):
        self.mode = mode
        self.device = "cuda"
        self.input_size = 256
        self.radius = 5
        self.hidden_size = 512
        self.input_shape = (self.input_size, self.input_size)
        self.mpp = 0.1
        self.setup_networks(self.input_shape)

        LR = 2.5e-4
        self.optimizer = optim.Adam(
            list(self.model.parameters()),
            lr=LR,
        )
        self.batch_length = 8

        self.tb_dir = TB_DIR
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)
        self.n_iter = 20000
        self.n_epochs = 20
        print("loading train")
        if mode != "test":
            self.train_dataloader = self.load_data(
                MAP_PTH,
                GOAL_PTH,
                WPT_PTH,
                batch_size=self.batch_length
                # EVAL_MAP_PTH,
                # EVAL_GOAL_PTH,
                # EVAL_WPT_PTH,
                # batch_size=self.batch_length,
            )
            print("loading val")
            self.val_dataloader = self.load_data(
                EVAL_MAP_PTH, EVAL_GOAL_PTH, EVAL_WPT_PTH, batch_size=1
            )
            print("# train", len(self.train_dataloader))
            print("# val: ", len(self.val_dataloader))

    def load_data(self, map_pth, goal_pth, wpt_pth, batch_size, shuffle=True):
        input = self.setup_input(map_pth, goal_pth)
        output = self.setup_output(wpt_pth)
        dataset = tuple(zip(input, output))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def setup_input(self, map_pth, goal_pth):
        return np.load(map_pth).astype(np.float32)

        # context_maps = np.load(map_pth).astype(np.float32)
        # print(" # maps: ", len(context_maps))
        # context_goals = np.load(goal_pth).astype(np.float32)
        # print(" # goals: ", len(context_goals))

        # return np.concatenate(
        #     [
        #         np.expand_dims(context_maps, axis=1),
        #         np.expand_dims(context_goals, axis=1),
        #     ],
        #     axis=1,
        # )

    def setup_output(self, wpt_pth):
        context_waypoints = np.load(wpt_pth).astype(np.float32)
        return np.array(
            [
                np.exp(context_waypoints[:, 0]),
                np.sin(context_waypoints[:, 1]),
                np.cos(context_waypoints[:, 1]),
            ]
        ).T

    def setup_networks(self, input_shape):
        self.model = Encoder(input_shape, self.hidden_size).to(self.device)

    def compute_r_theta(self, goal_coord):
        origin = np.array([self.input_shape[0] // 2, self.input_shape[0] // 2])

        r = np.linalg.norm(origin - goal_coord)
        theta = np.arctan2(
            goal_coord[1] - origin[1], goal_coord[0] - origin[0]
        )

        return torch.tensor(
            [r, np.sin(theta), np.cos(theta)],
            dtype=torch.float,
            device=self.device,
        )

    def r_theta_to_coord(self, r_theta):
        r, sin_theta, cos_theta = r_theta[0]
        theta = np.arctan2(sin_theta, cos_theta)
        x = (r / self.mpp) * np.cos(theta)
        y = (r / self.mpp) * np.sin(theta)
        mid = self.input_shape[0] // 2
        row, col = np.clip(
            int(mid - x), 0 + self.radius, self.input_size - self.radius
        ), np.clip(
            int(mid - y), 0 + self.radius, self.input_size - self.radius
        )
        return row, col
        # return x, y

    def train(self):
        batch_num = 0.0
        train_loss = 0.0
        for e in range(self.n_epochs):
            for input_map, label_wpt_vec in self.train_dataloader:
                pred_out = self.model(input_map.to(self.device))
                loss = F.mse_loss(pred_out, label_wpt_vec.to(self.device))
                self.optimizer.zero_grad()
                loss /= float(self.batch_length)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                batch_num += 1
                if batch_num % 500 == 0:
                    val_loss = self.eval()
                    print(
                        f"# epochs: {e}, # batch: {batch_num}, train_loss: {train_loss / batch_num:.4f}, val_loss: {val_loss.item():.4f}"
                    )

                    if self.writer is not None:
                        loss_data = {
                            "train_loss": train_loss / batch_num,
                            "val_loss": val_loss,
                        }
                        self.writer.add_scalars("loss", loss_data, batch_num)
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(
                            MODEL_DIR,
                            f"planning_model_50k_{batch_num}_{val_loss:.4f}.pth",
                        ),
                    )

    def eval(self):
        val_loss = 0
        n_val = len(self.val_dataloader)
        with torch.no_grad():
            for input_map, label_wpt_vec in self.val_dataloader:
                pred_out = self.model(input_map.to(self.device))
                val_loss += F.mse_loss(pred_out, label_wpt_vec.to(self.device))
        return val_loss / n_val

    def test(self, model_pth):
        self.model.load_state_dict(torch.load(os.path.abspath(model_pth)))
        self.model.eval()
        self.batch_length = 1
        TEST_MAP_PTH = os.path.join(
            BASE_PTH, "data/1157_data/context_maps_1157.npy"
        )
        TEST_GOAL_PTH = os.path.join(
            BASE_PTH, "data/1157_data/context_goals_1157.npy"
        )
        TEST_WPT_PTH = os.path.join(
            BASE_PTH, "data/1157_data/context_waypoints_1157.npy"
        )
        # TEST_MAP_PTH = os.path.join(BASE_PTH, "1157_data/context_maps.npy")
        # TEST_GOAL_PTH = os.path.join(BASE_PTH, "1157_data/context_goals.npy")
        # TEST_WPT_PTH = os.path.join(
        #     BASE_PTH, "1157_data/context_waypoints.npy"
        # )

        self.test_dataloader = self.load_data(
            TEST_MAP_PTH,
            TEST_GOAL_PTH,
            TEST_WPT_PTH,
            batch_size=1,
            shuffle=False,
        )
        action_loss = 0
        i = 0
        with torch.no_grad():
            for input_map, label_wpt_vec in self.test_dataloader:
                pred_out = self.model(input_map.to(self.device))
                cat_imgs = []
                # input map
                cat_imgs.append(input_map[0, :, :, 0].cpu().numpy() * 255.0)
                cat_imgs.append(np.ones((self.input_size, 1)) * 255.0)

                # input curr position and goal
                cat_imgs.append(input_map[0, :, :, 1].cpu().numpy() * 255.0)
                cat_imgs.append(np.ones((self.input_size, 1)) * 255.0)

                # # predicted waypoint
                pred_img = np.zeros(self.input_shape)
                x, y = self.r_theta_to_coord(pred_out.detach().cpu().numpy())
                rr, cc = disk((x, y), self.radius)
                pred_img[rr, cc] = 1.0
                # cat_imgs.append(pred_img * 255.0)
                # cat_imgs.append(np.ones((self.input_size, 1)))

                # overlay
                overlay = input_map[0, :, :, 0].cpu().numpy().copy()
                overlay[input_map[0, :, :, 1] == 1] = 0.3
                overlay[pred_img == 1] = 0.7
                cat_imgs.append(overlay * 255.0)
                cat_imgs.append(np.ones((self.input_size, 1)) * 255.0)

                # overlay gt
                overlay_gt = input_map[0, :, :, 0].cpu().numpy().copy()
                overlay_gt[input_map[0, :, :, 1] == 1] = 0.3

                # gt waypoint
                gt_img = np.zeros(self.input_shape)
                x, y = self.r_theta_to_coord(
                    label_wpt_vec.detach().cpu().numpy()
                )
                # print("label: ", label_wpt_vec, "x: ", x, "y: ", y)

                rr, cc = disk((x, y), self.radius)
                gt_img[rr, cc] = 1.0

                overlay_gt[gt_img == 1] = 0.7
                cat_imgs.append(overlay_gt * 255.0)

                action_loss += F.mse_loss(
                    pred_out, label_wpt_vec.to(self.device)
                )
                print(
                    "pred: ",
                    pred_out[0],
                    pred_out[0][0],
                    torch.rad2deg(torch.atan2(pred_out[0][1], pred_out[0][2])),
                )
                print(
                    "label: ",
                    label_wpt_vec[0],
                    label_wpt_vec[0][0],
                    torch.rad2deg(
                        torch.atan2(label_wpt_vec[0][1], label_wpt_vec[0][2])
                    ),
                )

                cat_img = np.concatenate(cat_imgs, axis=1)
                cv2.imwrite(os.path.join(IMG_DIR, f"cat_{i}.png"), cat_img)
                i += 1
            print("action loss: ", action_loss / i)


if __name__ == "__main__":
    P = Planner(sys.argv[1])
    if sys.argv[1] == "train":
        print("training!")
        P.train()
    elif sys.argv[1] == "eval":
        print("evaluating!")
        P.eval()
    elif sys.argv[1] == "test":
        print("evaluating!")
        P.test(sys.argv[2])
    else:
        print("please specify train or eval")
