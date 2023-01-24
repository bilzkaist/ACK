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



save_dir = cwd + '/data/output/site1/F1'
path_image_save_dir = save_dir + '/path_images'
step_position_image_save_dir = save_dir
magn_image_save_dir = save_dir
wifi_image_save_dir = save_dir + '/wifi_images'
ibeacon_image_save_dir = save_dir + '/ibeacon_images'
wifi_count_image_save_dir = save_dir

class AIKF(nn.Module):
    def init(self, input_size, hidden_size, output_size):
        super(AIKF, self).init()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_model(self, train_input, train_target, device, num_epochs, learning_rate):
        self.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        train_dataset = TensorDataset(train_input, train_target)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))
        return self


def calibrate_magnetic_wifi_ibeacon_to_position(path_file_list):
    location_model = None
    mwi_datas = {}
    for path_filename in path_file_list:
        print(f'Processing {path_filename}...')
        path_datas = read_data_file(path_filename)
        acce_datas = path_datas.acce
        magn_datas = path_datas.magn
        ahrs_datas = path_datas.ahrs
        wifi_datas = path_datas.wifi
        ibeacon_datas = path_datas.ibeacon
        posi_datas = path_datas.waypoint

        step_positions = compute_step_positions(acce_datas, ahrs_datas, posi_datas)
        # visualize_trajectory(posi_datas[:, 1:3], floor_plan_filename, width_meter, height_meter, title='Ground Truth', show=True)
        # visualize_trajectory(step_positions[:, 1:3], floor_plan_filename, width_meter, height_meter, title='Step Positions')
        # save_figure_to_html(step_position_image_save_dir + '/' + os.path.basename(path_filename) + '_step_positions.html')
        # visualize_heatmap(posi_datas[:, 1:3], width_meter, height_meter, title='Ground Truth', show=True)
        # visualize_heatmap(step_positions[:, 1:3], width_meter, height_meter, title='Step Positions')
        # save_figure_to_html(step_position_image_save_dir + '/' + os.path.basename(path_filename) + '_heatmap_step_positions.html')

        # create a dictionary of magnetic, wifi, ibeacon data and step positions
        mwi_datas[path_filename] = {
            'magn': magn_datas,
            'wifi': wifi_datas,
            'ibeacon': ibeacon_datas,
            'step': step_positions
        }

        # split magnetic, wifi, ibeacon data into fixed-length sequences
        magn_seqs, _ = split_ts_seq(magn_datas, seq_len=60, step_len=20)
        wifi_seqs, _ = split_ts_seq(wifi_datas, seq_len=60, step_len=20)
        ibeacon_seqs, _ = split_ts_seq(ibeacon_datas, seq_len=60, step_len=20)

        # count the number of occurrences of each wifi and ibeacon within each sequence
        wifi_counts = np.zeros((wifi_seqs.shape[0], wifi_seqs.shape[2] + 1))
        wifi_counts[:, 0] = wifi_seqs[:, 0, 0]
        for i in range(wifi_seqs.shape[0]):
            for j in range(wifi_seqs.shape[1]):
                wifi_counts[i, wifi_seqs[i, j, 1]] += 1

        ibeacon_counts = np.zeros((ibeacon_seqs.shape[0], len(ibeacon_seqs.unique_ibeacons) + 1))
        ibeacon_counts[:, 0] = ibeacon_seqs[:, 0, 0]
        for i in range(ibeacon_seqs.shape[0]):
            for j in range(ibeacon_seqs.shape[1]):
                ibeacon_counts[i, ibeacon_seqs[i, j, 1]] += 1

        # Normalize counts
        wifi_counts[:, 1:] = wifi_counts[:, 1:] / np.max(wifi_counts[:, 1:], axis=0)
        ibeacon_counts[:, 1:] = ibeacon_counts[:, 1:] / np.max(ibeacon_counts[:, 1:], axis=0)

        # split step positions into training and test sets
        train_inputs, test_inputs, train_targets, test_targets = split_ts_seq(np.hstack((magn_datas[:, 1:4], wifi_counts[:, 1:], ibeacon_counts[:, 1:])), step_positions[:, 1:3], seq_len=60, step_len=20, test_ratio=0.1)

        # define the model, loss function and optimizer
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_size = train_inputs.shape[1]
        hidden_size = 100
        output_size = 2
        location_model = AIKF(input_size, hidden_size, output_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(location_model.parameters(), lr=learning_rate)
        
        # train the model
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = location_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))
        
        # save the model
        torch.save(location_model.state_dict(), save_dir + '/' + os.path.basename(path_filename) + '_location_model.pt')
        
        # print the final loss
        with torch.no_grad():
            test_inputs = np.hstack((magn_datas[:, 1:4], wifi_counts[:, 1:], ibeacon_counts[:, 1:]))
            test_targets = step_positions[:, 1:3]
            test_inputs = torch.from_numpy(test_inputs).float().to(device)
            test_targets = torch.from_numpy(test_targets).float().to(device)
            test_outputs = location_model(test_inputs)
            test_loss = criterion(test_outputs, test_targets)
            print("Final test loss: ", test_loss)
    return location_model





def test2():
    with open(floor_info_filename) as json_file:
        floor_info = json.load(json_file)
    width_meter = floor_info["map_info"]["width"]
    height_meter = floor_info["map_info"]["height"]
    test_path_filename = None #'./data/site1/F1/path_data_files/path_1.json'
    test_path_filenames = list(Path(path_data_dir).resolve().glob("*.txt"))

    # 1. visualize ground truth positions
    print('Visualizing ground truth positions...')
    for test_path_filename in test_path_filenames:
        print(f'Processing file: {test_path_filename}...')

        path_data = read_data_file(test_path_filename)
        path_id = test_path_filename.name.split(".")[0]
        fig = visualize_trajectory(path_data.waypoint[:, 1:3], floor_plan_filename, width_meter, height_meter, title=path_id, show=False)
        html_filename = f'{path_image_save_dir}/{path_id}.html'
        html_filename = str(Path(html_filename).resolve())
        print("html_filename = ",html_filename)
        save_figure_to_html(fig, html_filename)


    test_path_datas = read_data_file(test_path_filename)
    test_magn_datas = test_path_datas.magn
    test_wifi_datas = test_path_datas.wifi
    test_ibeacon_datas = test_path_datas.ibeacon

    test_magn_counts = split_ts_seq(test_magn_datas, 10)
    test_wifi_counts = split_ts_seq(test_wifi_datas, 10)
    test_ibeacon_counts = split_ts_seq(test_ibeacon_datas, 10)

    test_inputs = np.hstack((test_magn_counts[:, 1:4], test_wifi_counts[:, 1:], test_ibeacon_counts[:, 1:]))

    test_inputs = torch.from_numpy(test_inputs).float()
    test_inputs = test_inputs.to(device)

    test_outputs = location_model(test_inputs)

    test_outputs = test_outputs.data.cpu().numpy()

    visualize_trajectory(test_outputs, floor_plan_filename, width_meter, height_meter, title='Predicted', show=True)
    save_figure_to_html(test_outputs, floor_plan_filename, width_meter, height_meter, title='Predicted', save_dir=path_image_save_dir, show=True)

    print("Testing Done!")


def test():
    with open(floor_info_filename) as json_file:
        floor_info = json.load(json_file)
    width_meter = floor_info["map_info"]["width"]
    height_meter = floor_info["map_info"]["height"]
    test_path_filename = None 
    test_path_filenames = list(Path(path_data_dir).resolve().glob("*.txt"))

    # 1. visualize ground truth positions
    print('Visualizing ground truth positions...')
    for test_path_filename in test_path_filenames:
        print(f'Processing file: {test_path_filename}...')

        path_data = read_data_file(test_path_filename)
        #print(path_data)
        path_id = test_path_filename.name.split(".")[0]
        fig = visualize_trajectory(path_data.waypoint[:, 1:3], floor_plan_filename, width_meter, height_meter, title=path_id, show=False)
        html_filename = f'{path_image_save_dir}/{path_id}.html'
        html_filename = str(Path(html_filename).resolve())
        print("html_filename = ",html_filename)
        save_figure_to_html(fig, html_filename)
        
    test_path_datas = read_data_file(test_path_filename)
    test_magn_datas = test_path_datas.magn
    test_wifi_datas = test_path_datas.wifi
    test_ibeacon_datas = test_path_datas.ibeacon

    
    test_magn_counts = split_ts_seq(test_magn_datas, 10)
    test_wifi_counts = split_ts_seq(test_wifi_datas, 10)
    test_ibeacon_counts = split_ts_seq(test_ibeacon_datas, 10) 
    
    
    test_magn_counts = np.array(test_magn_counts)
    test_wifi_counts = np.array(test_wifi_counts)
    test_ibeacon_counts = np.array(test_ibeacon_counts)

    test_inputs = np.hstack((test_magn_counts[:, 1], test_magn_counts[:, 2], test_magn_counts[:, 3], test_wifi_counts[:, 1], test_wifi_counts[:, 2], test_ibeacon_counts[:, 1]))

    test_inputs = []
    test_inputs.extend(test_magn_counts[:, 1])
    test_inputs.extend(test_magn_counts[:, 2])
    test_inputs.extend(test_magn_counts[:, 3])
    test_inputs.extend(test_wifi_counts[:, 1])
    test_inputs.extend(test_wifi_counts[:, 2])
    test_inputs.extend(test_ibeacon_counts[:, 1])

    test_inputs = np.array(test_inputs)



    """ 
    try:
        test_inputs = np.hstack((test_magn_counts[:, 1:4], test_wifi_counts[:, 1:], test_ibeacon_counts[:, 1:]))
    except:
        test_inputs = np.hstack((test_magn_counts[:, 1], test_magn_counts[:, 2], test_magn_counts[:, 3], test_wifi_counts[:, 1], test_wifi_counts[:, 2], test_ibeacon_counts[:, 1]))
    """
    #test_inputs = np.hstack((test_magn_counts[:, 1], test_magn_counts[:, 2], test_magn_counts[:, 3], test_wifi_counts[:, 1], test_wifi_counts[:, 2], test_ibeacon_counts[:, 1]))


    test_inputs = torch.from_numpy(test_inputs).float()
    test_inputs = test_inputs.to(device)

    test_outputs = location_model(test_inputs)

    test_outputs = test_outputs.data.cpu().numpy()

    fig = visualize_trajectory(test_outputs, floor_plan_filename, width_meter, height_meter, title='Predicted', show=True)
    save_figure_to_html(fig, f'{path_image_save_dir}/predicted.html')

    print("Testing Done!")



if __name__ == '__main__':
    try:
        path_file_list = [f for f in os.listdir(path_data_dir) if f.endswith('.json')]
    except:
        path_file_list = [f for f in os.listdir(path_data_dir) if f.endswith('.json')]
    
    with open(floor_info_filename, 'r') as f:
        floor_info = json.load(f)
    print(floor_info)
    
    
    calibrate_magnetic_wifi_ibeacon_to_position(path_file_list)#, learning_rate, num_epochs, )
    print("Done!")
    test()
    print("Testing Done!")
    print("Testing Done!")
    train_inputs = np.hstack((magn_datas[:, 1:4], wifi_counts[:, 1:], ibeacon_counts[:, 1:]))
    train_targets = step_positions[:, 1:3]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_size = train_inputs.shape[1]
    hidden_size = 100
    output_size = 2
    model = AIKF
        
