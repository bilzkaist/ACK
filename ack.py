import json
import os
from pathlib import Path
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from compute_f import split_ts_seq, compute_step_positions
from io_f import read_data_file
from visualize_f import visualize_trajectory, visualize_heatmap, save_figure_to_html

cwd = os.getcwd()
print("cwd = ", cwd)
floormappath = cwd + "\\data\\images\\floormapN17.jpeg"
resultspath = cwd + "\\data\\results\\"
print("floormappath = ",floormappath)

#floor_data_dir = './data/site1/F1'
floor_data_dir = "S:\\indoor-location-competition-20-master\\indoor-location-competition-20-master\\data\site1\\F1"
path_data_dir = floor_data_dir + '/path_data_files'
floor_plan_filename = floor_data_dir + '/floor_image.png'
floor_info_filename = floor_data_dir + '/floor_info.json'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

save_dir = cwd + '/data/output/site1/F1'
path_image_save_dir = save_dir + '/path_images'
step_position_image_save_dir = save_dir
magn_image_save_dir = save_dir
wifi_image_save_dir = save_dir + '/wifi_images'
ibeacon_image_save_dir = save_dir + '/ibeacon_images'
wifi_count_image_save_dir = save_dir

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, pf_dim):
        super(Transformer, self).__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(input_size, num_heads, pf_dim), num_layers)
        self.decoder = nn.Linear(hidden_size, 2) # 2 for x,y

    def forward(self, src, mask=None):
        src = self.encoder(src, mask)
        src = self.decoder(src)
        return src


model = Transformer(input_size, hidden_size, num_layers, num_heads, pf_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_data = TensorDataset(acce_datas, magn_datas, ahrs_datas, waypoints)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# training loop
for epoch in range(num_epochs):
    for i, (acce_batch, magn_batch, ahrs_batch, waypoints_batch) in enumerate(train_loader):
        acce_batch, magn_batch, ahrs_batch, waypoints_batch = acce_batch.to(device), magn_batch.to(device), ahrs_batch.to(device), waypoints_batch.to(device)
        pred = model(acce_batch, magn_batch, ahrs_batch)
        loss = F.mse_loss(pred, waypoints_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def load_data(path_file_list):
    print("load_data ...")
    location_model = None
    mwi_datas = {}
    path_datas = None
    acce_datas = None
    magn_datas = None
    ahrs_datas = None
    wifi_datas = None
    ibeacon_datas = None
    posi_datas = None

    input_data = []
    target_data = []

    for path_filename in path_file_list:
        print(f'Processing {path_filename}...')
        path_datas = read_data_file(path_filename)
        acce_datas = path_datas.acce
        magn_datas = path_datas.magn
        ahrs_datas = path_datas.ahrs
        wifi_datas = path_datas.wifi
        ibeacon_datas = path_datas.ibeacon
        posi_datas = path_datas.waypoint
        waypoints = path_datas.waypoint[:, 1:3]
        print("Target -> waypoints = ",waypoints)
        # split and count the sensor data
        acce_counts = split_ts_seq(acce_datas, 10)
        magn_counts = split_ts_seq(magn_datas, 10)
        ahrs_counts = split_ts_seq(ahrs_datas, 10)
        print("acce_counts = ", acce_counts)
        print("magn_counts = ", magn_counts)
        print("ahrs_counts = ", ahrs_counts)

        #print(" ", wifi_counts[:, 1], wifi_counts[:, 2], ibeacon_counts[:, 1])
    path_data_files = list(Path(path_data_dir).resolve().glob("*.txt"))
    
    #step_positions = compute_step_positions(path_data_files, ahrs_datas, posi_datas)
    step_positions = compute_step_positions(acce_datas, ahrs_datas, posi_datas)
    
    # create a dictionary of magnetic, wifi, ibeacon data and step positions
    mwi_datas[path_filename] = {
        'magn': magn_datas,
        'wifi': wifi_datas,
        'ibeacon': ibeacon_datas,
        'step': step_positions
    }

    train_magn_datas = []
    train_wifi_datas = []
    train_ibeacon_datas = []
    train_target = []

    for path_data_file in path_data_files:
        path_datas = read_data_file(path_data_file)
        train_magn_datas.extend(path_datas.magn)
        train_wifi_datas.extend(path_datas.wifi)
        train_ibeacon_datas.extend(path_datas.ibeacon)
        train_target.extend(step_positions[path_data_file.name])

    train_magn_counts = split_ts_seq(train_magn_datas, 10)
    train_wifi_counts = split_ts_seq(train_wifi_datas, 10)
    train_ibeacon_counts = split_ts_seq(train_ibeacon_datas, 10)

    input_data = np.hstack((train_magn_counts[:, 1], train_magn_counts[:, 2], train_magn_counts[:, 3], train_wifi_counts[:, 1], train_wifi_counts[:, 2], train_ibeacon_counts[:, 1]))
    input_data = torch.from_numpy(input_data).float()
    target_data = torch.from_numpy(np.array(train_target)).float()

    return input_data, target_data


""" def get_path_data_files():
    for path_filename in path_file_list:
        print(f'Processing {path_filename}...')
        path_datas = read_data_file(path_filename)
        acce_datas = path_datas.acce
        magn_datas = path_datas.magn
        ahrs_datas = path_datas.ahrs
        wifi_datas = path_datas.wifi
        ibeacon_datas = path_datas.ibeacon
        posi_datas = path_datas.waypoint
    path_data_files = list(Path(path_data_dir).resolve().glob("*.txt"))
    return path_data_files """


def run():
    print("AIKF  Started -!!!")
    # Enter code here
    input_data = []
    target_data = []
    path_file_list = list(Path(path_data_dir).resolve().glob("*.txt"))
    input_data, target_data = load_data(path_file_list)
    print("AIKF Finished Successfully !!!")


if __name__ == '__main__':
    print("Program  Started !!!")
    run()
    print("Program Finished Successfully !!!")
