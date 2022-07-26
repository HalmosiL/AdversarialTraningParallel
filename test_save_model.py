from ModelLodaer import resnet_slice_model, get_resnet18_hourglass
import torch

model = get_resnet18_hourglass("cpu", encoder_weights=None)
torch.save(model.state_dict(), "./model_cache/model_1.pt")
