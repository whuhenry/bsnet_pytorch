import argparse
import torch
from dataset.single_pixel import SinglePixelDataset
from torch.utils.data.dataloader import DataLoader
from bs_net_fc import BSNetFC
import torch.optim as optim
import torch.optim.lr_scheduler

def bs_net_fc_loss(input_x, weight, reconstruced_x, lamda=0.01) -> torch.Tensor:
    recontruction_loss = torch.mean((reconstruced_x - input_x) ** 2)
    regularization = lamda * torch.sum(weight)

    return recontruction_loss + regularization


def main():
    parser = argparse.ArgumentParser(description='Parameters.')
    parser.add_argument('-dataset', type=str, help='input dataset path')
    parser.add_argument('-epoch', type=int, help='train epoch')
    parser.add_argument('-batch', type=int, help='batch size')

    args = parser.parse_args()

    train_dataset = SinglePixelDataset(args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BSNetFC(train_dataset.bands)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_func = bs_net_fc_loss

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.5)

    torch.save(model.state_dict(), 'pretrained/model-{1}.pt')

    model.train()

    for epoch in range(args.epoch):
        for batch_idx, input_x in enumerate(train_loader, 0):
            input_x = input_x.to(device)
            weight, rec_x = model(input_x)

            loss = loss_func(input_x, weight, rec_x, 0.01)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"epoch: {epoch}, batch : {batch_idx}, loss : { loss.item() }")
        scheduler.step()

    torch.save(model.state_dict(), 'pretrained/model-{1}.pt')


if __name__ == "__main__":
    main()
