import argparse
from bs_net_fc import BSNetFC
from dataset.single_pixel import SinglePixelDataset
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters.')
    parser.add_argument('-dataset', type=str, help='input dataset path')
    parser.add_argument('-model', type=str, help='pretrained model')

    args = parser.parse_args()

    eval_dataset = SinglePixelDataset(args.dataset)
    eval_data = torch.FloatTensor(np.reshape(eval_dataset.data_array, [eval_dataset.bands, -1]).T)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BSNetFC(eval_dataset.bands)
    model.load_state_dict(torch.load(args.model))
    model.to(device)
    model.eval()

    weight, rec_x = model(eval_data)

    print(torch.mean((rec_x - eval_data) ** 2).item())
    
    all_weight = np.sum(weight.cpu().detach().numpy(), axis=0)
    print(np.argsort(all_weight))