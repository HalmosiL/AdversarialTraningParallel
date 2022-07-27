from ModelLodaer import resnet_slice_model, get_resnet18_hourglass
from Cityscapes import CitySegmentation
from Adversarial import Cosine_PDG_Adam
from Adversarial import model_immer_attack_auto_loss
import torchvision.transforms as T
import torch

import sys
import os
import time

def run(id_, batch, device, model, attack, number_of_steps, data_queue):
    print("Gen_", id_, " started..")

    image = batch[0].to(device)
    label = batch[1].to(device)

    image = model_immer_attack_auto_loss(
        image=image,
        model=model,
        attack=attack,
        number_of_steps=number_of_steps,
        device=device
    )

def start(device):
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
        get_resnet18_hourglass(device, encoder_weights=None),
        level="Encoder"
    )

    attack = Cosine_PDG_Adam(
        step_size=1,
        clip_size=0.02,
        reset_period=2,
        batch_size=1
    )
    
    run(0, batch, device, model, attack, 2, "./")

if __name__ == '__main__':
    GPU_MAX_memory_in_used = 4

    def get_freer_gpu(gpu_id=None):
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >xtmp')
        memory_used = [int(x.split()[2]) for x in open('xtmp', 'r').readlines()]
        
        print(memory_used)

        free_gpu = []

        if(gpu_id):
            if(GPU_MAX_memory_in_used >= memory_used[gpu_id]):
                free_gpu.append(gpu_id)
        else:
            for i in range(len(memory_used)):
                if(GPU_MAX_memory_in_used >= memory_used[i]):
                    free_gpu.append(i)

        return free_gpu

    print("Create folder models...")

    try:
        os.mkdir("./models")
        print("Folder created successfully...")
    except:
        print("Folder alredy exists...")


    free_gpu = get_freer_gpu()

    while(not len(free_gpu)):
        print("There is no free GPU wait 10(s)...")
        time.sleep(10)
        free_gpu = get_freer_gpu()


    if(len(free_gpu)):
        device = 'cuda:' + str(free_gpu[0])
        print("Use GPU:" + device)
        start(device)
