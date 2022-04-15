import torch

from pathlib import Path

DATA_PATH = Path("data/")

def save_dataset(data_x, data_y, type, size):
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    tensor_x = torch.from_numpy(data_x).float()
    tensor_x = tensor_x.permute(0, 3, 1, 2).contiguous()
    tensor_y = torch.from_numpy(data_y).long()

    torch.save(tensor_x, DATA_PATH / f"{type}_data_{size}.pt")
    torch.save(tensor_y, DATA_PATH / f"{type}_label_{size}.pt")

def load_dataset(type, size):
    tensor_x = torch.load(DATA_PATH / f"{type}_data_{size}.pt")
    nparray_x = tensor_x.round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    tensor_y = torch.load(DATA_PATH / f"{type}_label_{size}.pt")
    nparray_y = tensor_y.numpy()

    return (nparray_x, nparray_y)
