from ModelLodaer import resnet_slice_model, get_resnet18_hourglass
from Cityscapes import CitySegmentation
from Adversarial import Cosine_PDG_Adam
from Adversarial import model_immer_attack_auto_loss
import torchvision.transforms as T
import torch

def run(id_, batch, device, model, attack, number_of_steps, data_queue):
    print("Gen_", id_, " started..")
    model = model.to(device)

    image = batch[0].to(device)
    label = batch[1].to(device)

    image = model_immer_attack_auto_loss(
        image=image,
        model=model,
        attack=attack,
        number_of_steps=number_of_steps,
        device=device
    )

def start():
    input_transform = T.Compose([
        T.ToTensor(),
    ])

    train_data_set = CitySegmentation(
        root="../../../../Data/City_scapes_data/",
        split="train",
        transform=input_transform,
        start_index=0,
        end_index=50
    )

    train_data_set_loader = torch.utils.data.DataLoader(
        train_data_set,
        batch_size=16,
        num_workers=0
    )

    batch = next(iter(train_data_set_loader))

    model = resnet_slice_model(
        get_resnet18_hourglass("cpu", encoder_weights=None),
        level="Encoder"
    )

    attack = Cosine_PDG_Adam(
        step_size=1,
        clip_size=0.02,
        reset_period=2,
        batch_size=1
    )
    
    run(0, batch, "cuda:0", model, attack, 2, "./")

if __name__ == '__main__':
    start()
