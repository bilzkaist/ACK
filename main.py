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


floor_data_dir = './data/site1/F1'
path_data_dir = floor_data_dir + '/path_data_files'
floor_plan_filename = floor_data_dir + '/floor_image.png'
floor_info_filename = floor_data_dir + '/floor_info.json'

save_dir = './output/site1/F1'
path_image_save_dir = save_dir + '/path_images'
step_position_image_save_dir = save_dir
magn_image_save_dir = save_dir
wifi_image_save_dir = save_dir + '/wifi_images'
ibeacon_image_save_dir = save_dir + '/ibeacon_images'
wifi_count_image_save_dir = save_dir


class LocationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LocationModel, self).__init__()
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
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

def calibrate_magnetic_wifi_ibeacon_to_position(path_file_list):
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
    # visualize_trajectory(step_positions[:, 1:3], floor_plan_filename, width_meter, height_meter, title='Step Position', show=True)

    if wifi_datas.size != 0:
        sep_tss = np.unique(wifi_datas[:, 0].astype(float))
    wifi_seqs = split_ts_seq(wifi_datas, sep_tss)
    wifi_counts = []
    for wifi_seq in wifi_seqs:
        wifi_counts.append(len(np.unique(wifi_seq[:, 2])))
        wifi_counts = np.array(wifi_counts)
        wifi_counts = wifi_counts.reshape(-1, 1)
        wifi_counts = np.hstack((sep_tss[:-1].reshape(-1, 1), wifi_counts))
        visualize_heatmap(wifi_counts[:, 1], floor_plan_filename, width_meter, height_meter, title='Wifi Counts')
        save_figure_to_html(wifi_count_image_save_dir + '/wifi_counts.html', dpi=100)

     # calibrate ibeacon
    if ibeacon_datas.size != 0:
        ibeacon_datas[:, 1] = [str(int(i)) for i in ibeacon_datas[:, 1]]
        ibeacon_datas = ibeacon_datas[ibeacon_datas[:, 1].argsort()]
        sep_tss = np.unique(ibeacon_datas[:, 0].astype(float))
        ibeacon_seqs = split_ts_seq(ibeacon_datas, sep_tss)
        ibeacon_counts = []
        for ibeacon_seq in ibeacon_seqs:
            ibeacon_counts.append(len(np.unique(ibeacon_seq[:, 1])))
        ibeacon_counts = np.array(ibeacon_counts)
        ibeacon_counts = ibeacon_counts.reshape(-1, 1)
        ibeacon_counts = np.hstack((sep_tss[:-1].reshape(-1, 1), ibeacon_counts))
        visualize_heatmap(ibeacon_counts[:, 1], floor_plan_filename, width_meter, height_meter, title='IBeacon Counts')
        save_figure_to_html(ibeacon_image_save_dir + '/ibeacon_counts.html', dpi=100)

    # train model

    if step_positions.size != 0:
        train_inputs = np.hstack((magn_datas[:, 1:4], wifi_counts[:, 1:], ibeacon_counts[:, 1:]))
        train_targets = step_positions[:, 1:3]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LocationModel(input_size=train_inputs.shape[1], hidden_size=64, output_size=2)
        train_model(model, train_inputs, train_targets, device, num_epochs=50, learning_rate=0.001)

        torch.save(model.state_dict(), f'{save_dir}/location_model.pth')
        print(f"Model is saved to {save_dir}/location_model.pth")
        
        # test model
        test_inputs = train_inputs
        test_targets = train_targets
        test_outputs = model(torch.Tensor(test_inputs).to(device))
        test_outputs = test_outputs.cpu().detach().numpy()
        
        visualize_trajectory(test_targets, floor_plan_filename, title='Ground Truth', save_filename=f'{save_dir}/ground_truth.html')
        visualize_trajectory(test_outputs, floor_plan_filename, title='Prediction', save_filename=f'{save_dir}/prediction.html')
        visualize_heatmap(test_targets, floor_plan_filename, title='Ground Truth', save_filename=f'{save_dir}/ground_truth_heatmap.html')
        visualize_heatmap(test_outputs, floor_plan_filename, title='Prediction', save_filename=f'{save_dir}/prediction_heatmap.html')
        
        # save figures to html
        save_figure_to_html(f'{save_dir}/ground_truth.html', f'{save_dir}/ground_truth.html')
        save_figure_to_html(f'{save_dir}/prediction.html', f'{save_dir}/prediction.html')
        save_figure_to_html(f'{save_dir}/ground_truth_heatmap.html', f'{save_dir}/ground_truth_heatmap.html')
        save_figure_to_html(f'{save_dir}/prediction_heatmap.html', f'{save_dir}/prediction_heatmap.html')
        
        print('Calibration is done.')
    else:
        print('No step positions are found.')
    
    return model

if name == 'main':
    with open(floor_info_filename, 'r') as f:
    floor_info = json .load(f)
    width_meter = floor_info['width_meter']
    height_meter = floor_info['height_meter']

    path_file_list = [path_data_dir + '/' + f for f in os.listdir(path_data_dir) if f.endswith('.json')]

    mwi_datas = calibrate_magnetic_wifi_ibeacon_to_position(path_file_list)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hidden_size = 100
    input_size = 7
    output_size = 2

    model = LocationModel(input_size, hidden_size, output_size)

    # train model
    if step_positions.size != 0:
        train_inputs = np.hstack((magn_datas[:, 1:4], wifi_counts[:, 1:], ibeacon_counts[:, 1:]))
        train_targets = step_positions[:, 1:3]
        train_model(model, train_inputs, train_targets, device, num_epochs=50, learning_rate=0.001)
        torch.save(model.state_dict(), './location_model.pt')
    else:
        print("Not enough data to train model")

    # Test
    test_inputs = np.hstack((magn_datas[:, 1:4], wifi_counts[:, 1:], ibeacon_counts[:, 1:]))
    test_outputs = model(torch.Tensor(test_inputs).to(device))
    test_outputs = test_outputs.detach().cpu().numpy()
    visualize_trajectory(test_outputs, floor_plan_filename, width_meter, height_meter, title='Predicted', show=True)
    visualize_trajectory(step_positions[:, 1:3], floor_plan_filename, width_meter, height_meter, title='Ground Truth', show=True)
    visualize_heatmap(test_outputs, floor_plan_filename, width_meter, height_meter, title='Predicted', save_filename=step_position_image_save_dir+'/predicted_heatmap.html')
    visualize_heatmap(step_positions[:, 1:3], floor_plan_filename, width_meter, height_meter, title='Ground Truth', save_filename=step_position_image_save_dir+'/ground_truth_heatmap.html')
    print("Testing Done!")

if name == 'main':
    path_file_list = [f'{path_data_dir}/{f}' for f in os.listdir(path_data_dir) if f.endswith('.txt')]
    print("Calibrating magnetic, wifi, and ibeacon data to position...")
    calibrate_magnetic_wifi_ibeacon_to_position(path_file_list)
    print("Calibration Done!")

    print("Testing the model...")
    test_inputs = np.hstack((magn_datas[:, 1:4], wifi_counts[:, 1:], ibeacon_counts[:, 1:]))
    test_targets = step_positions[:, 1:3]
    test_inputs, test_targets = torch.tensor(test_inputs, dtype=torch.float), torch.tensor(test_targets, dtype=torch.float)
    test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
    with torch.no_grad():
        test_outputs = model(test_inputs)
        test_loss = criterion(test_outputs, test_targets)
    print('Test loss: %.3f' % test_loss)
    print("Testing Done!")

    # Save the model and calibration data
    torch.save(model.state_dict(), f'{save_dir}/location_model.pth')
    np.save(f'{save_dir}/mwi_datas.npy', mwi_datas)
    # Visualize the test results
    visualize_trajectory(test_targets, floor_plan_filename, width_meter, height_meter, title='Ground Truth', save_filename=f'{save_dir}/ground_truth.html')
    visualize_trajectory(test_outputs, floor_plan_filename, width_meter, height_meter, title='Predicted Trajectory', save_filename=f'{save_dir}/predicted_trajectory.html')
    visualize_heatmap(test_targets, floor_plan_filename, width_meter, height_meter, title='Ground Truth', save_filename=f'{save_dir}/ground_truth_heatmap.html')
    visualize_heatmap(test_outputs, floor_plan_filename, width_meter, height_meter, title='Predicted Trajectory', save_filename=f'{save_dir}/predicted_trajectory_heatmap.html')


if name == 'main':
    path_file_list = [f'{path_data_dir}/{f}' for f in os.listdir(path_data_dir) if f.endswith('.txt')]

    print("Calibrating magnetic, wifi, and ibeacon data to position...")
    calibrate_magnetic_wifi_ibeacon_to_position(path_file_list)
    print("Calibration Done!")

    print("Testing the model...")
    test_inputs = np.hstack((magn_datas[:, 1:4], wifi_counts[:, 1:], ibeacon_counts[:, 1:]))
    test_targets = step_positions[:, 1:3]
    test_inputs, test_targets = torch.tensor(test_inputs, dtype=torch.float), torch.tensor(test_targets, dtype=torch.float)
    test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
    with torch.no_grad():
        test_outputs = model(test_inputs)
        test_loss = criterion(test_outputs, test_targets)
    print('Test loss: %.3f' % test_loss)
    print("Testing Done!")

    # Save the model and calibration data
    torch.save(model.state_dict(), f'{save_dir}/location_model.pth')
    np.save(f'{save_dir}/mwi_datas.npy', mwi_datas)
    # Visualize the test results
    visualize_trajectory(test_targets, floor_plan_filename, width_meter, height_meter, title='Ground Truth', save_filename=f'{save_dir}/ground_truth.html')
    visualize_trajectory(test_outputs, floor_plan_filename, width_meter, height_meter, title='Predicted Trajectory', save_filename=f'{save_dir}/predicted_trajectory.html')
    visualize_heatmap(test_targets, floor_plan_filename, width_meter, height_meter, title='Ground Truth', save_filename=f'{save_dir}/ground_truth_heatmap.html')
    visualize_heatmap(test_outputs, floor_plan_filename, width_meter, height_meter, title='Predicted Trajectory', save_filename=f'{save_dir}/predicted_trajectory_heatmap.html')

    # Save the model
    model_save_path = save_dir + '/location_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

if name == 'main':
    path_file_list = [f for f in os.listdir(path_data_dir) if f.endswith('.json')]
    calibrate_magnetic_wifi_ibeacon_to_position(path_file_list)
    print("Done!")
    test()
    print("Testing Done!")

    # Save the trained model
    torch.save(model.state_dict(), save_dir + '/trained_location_model.pt')

    # Save the floor information
    with open(floor_info_filename, 'w') as f:
    json.dump({'width_meter': width_meter, 'height_meter': height_meter}, f)

    # Save the step position image
    step_position_image_filename = step_position_image_save_dir + '/step_position.html'
    save_figure_to_html(step_position_image_filename, width_meter, height_meter)

    # Save the magnetic heatmap image
    magn_image_filename = magn_image_save_dir + '/magn_heatmap.html'
    visualize_heatmap(magn_datas[:, 1:4], step_positions[:, 1:3], width_meter, height_meter, title='Magnetic Heatmap', save_filename=magn_image_filename)

    # Save the wifi heatmap image
    wifi_image_filename = wifi_image_save_dir + '/wifi_heatmap.html'
    visualize_heatmap(wifi_counts[:, 1:], step_positions[:, 1:3], width_meter, height_meter, title='Wi-Fi Heatmap', save_filename=wifi_image_filename)

    # Save the ibeacon heatmap image
    ibeacon_image_filename = ibeacon_image_save_dir + '/ibeacon_heatmap.html'
    visualize_heatmap(ibeacon_counts[:, 1:], step_positions[:, 1:3], width_meter, height_meter, title='iBeacon Heatmap', save_filename=ibeacon_image_filename)

    # Save the wifi count heatmap image
    wifi_count_image_filename = wifi_count_image_save_dir + '/wifi_count_heatmap.html'
    visualize_heatmap(wifi_counts[:, 1:], step_positions[:, 1:3], width_meter, height_meter, title='Wi-Fi Count Heatmap', save_filename=wifi_count_image_filename)

    print("All outputs have been saved successfully.")



