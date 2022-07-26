import torchvision.transforms as T
import glob
import time
import torch
import os

from Gen import Gen
from Cityscapes import CitySegmentation
from Adversarial import Cosine_PDG_Adam
from ModelLodaer import resnet_slice_model, get_resnet18_hourglass

class Executor:
    def __init__(
        self,
        model_cache,
        max_number_of_gens,
        queue_size_train,
        queue_size_val, 
        data_queue, 
        data_path, 
        batch_size, 
        number_of_steps,
        data_set_start_index_train,
        data_set_end_index_train,
        data_set_start_index_val,
        data_set_end_index_val,
        device
    ):
        try: 
            print("Create data cache...")
            os.mkdir(data_queue)
            os.mkdir(data_queue[:-1] + "_val")
            print("Data cache created successfuly...")
        except OSError as error: 
            print("Data cache alredy exist...")  

        self.model_cache = model_cache
        self.max_number_of_gens = max_number_of_gens
        self.queue_size_train = queue_size_train
        self.queue_size_val = queue_size_val
        self.data_path = data_path
        self.model_name = None
        self.number_of_gens = 0
        self.batch_size = batch_size
        self.device = device
        self.num_workers = 0
        self.number_of_steps = number_of_steps
        self.data_queue = data_queue

        self.data_set_start_index_train = data_set_start_index_train
        self.data_set_end_index_train = data_set_end_index_train
        self.data_set_start_index_val = data_set_start_index_val
        self.data_set_end_index_val = data_set_end_index_val
        
        input_transform = T.Compose([
            T.ToTensor(),
        ])

        self.train_data_set = CitySegmentation(
            root=self.data_path,
            split="train",
            transform=input_transform,
            start_index=data_set_start_index_train,
            end_index=data_set_end_index_train
        )

        self.val_data_set = CitySegmentation(
            root=self.data_path,
            split="val",
            transform=input_transform,
            start_index=data_set_start_index_val,
            end_index=data_set_end_index_val
        )

        self.train_data_set_loader = torch.utils.data.DataLoader(
            self.train_data_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

        self.val_data_set_loader = torch.utils.data.DataLoader(
            self.val_data_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

        if(data_set_end_index_train is None):
            self.train_data_set_len = train_data_set.__len__() - data_set_start_index_train
        else:
            self.train_data_set_len = data_set_end_index_train - data_set_start_index_train

        if(data_set_end_index_val is None):
            self.val_data_set_len = val_data_set.__len__() - data_set_start_index_val
        else:
            self.val_data_set_len = data_set_end_index_val - data_set_start_index_val

        self.train_element_id = 0
        self.val_element_id = 0

        self.attack = Cosine_PDG_Adam(
            step_size=1,
            clip_size=0.02,
            reset_period=number_of_steps,
            batch_size=batch_size
        )

    def start(self):
        train_iter = iter(self.train_data_set_loader)
        val_iter = iter(self.val_data_set_loader)

        train_id = 0
        val_id = 0

        id_k = 0

        while(True):
            new_model_name = glob.glob(self.model_cache + "*.pt")

            if(not len(new_model_name) and self.model_name is None):
                print("There is no model to use yet...")
                time.sleep(2)
            else:
                if(self.model_name != new_model_name[0]):
                    print("Use model:", new_model_name[0])
                    self.model_name = new_model_name[0]

                    model = resnet_slice_model(
                        get_resnet18_hourglass(self.device, encoder_weights=None),
                        level="Encoder"
                    )

                if(self.train_element_id < self.train_data_set_len):
                    number_elments_of_data_queue = len(glob.glob(self.data_queue + "/*"))

                    if(number_elments_of_data_queue <= self.queue_size_train * 2):
                        if(self.train_element_id == 0 and self.number_of_gens < self.max_number_of_gens):
                            print("Start generating traning data from:", self.data_set_start_index_train, " to:", self.data_set_end_index_train, "...")

                        self.number_of_gens = Gen.number_of_gens

                        if(self.number_of_gens < self.max_number_of_gens):
                            try:
                                self.train_element_id += 1
                                batch = next(train_iter)
                                train_id += 1

                                Gen(
                                    model,
                                    batch,
                                    train_id,
                                    self.attack,
                                    self.device,
                                    self.number_of_steps,
                                    self.data_queue
                                ).start()
                            except StopIteration:
                                train_iter = iter(self.train_data_set_loader)
                    else:
                        print("Data queue(Train) is full process is waiting...")
                        time.sleep(0.5)

                    self.val_element_id = 0
                else:
                    if(self.val_element_id < self.val_data_set_len):
                        if(self.val_element_id == 0 and self.number_of_gens < self.max_number_of_gens):
                            print("Start generating val data from:", self.data_set_start_index_val, " to:", self.data_set_end_index_val, "...")

                        number_elments_of_data_queue = len(glob.glob(self.data_queue[:-1] + "_val" + "/*"))

                        if(number_elments_of_data_queue <= self.queue_size_val * 2):
                            self.number_of_gens = Gen.number_of_gens

                            if(self.number_of_gens < self.max_number_of_gens):
                                try:
                                    self.val_element_id += 1
                                    batch = next(val_iter)
                                    val_id += 1

                                    Gen(
                                        model,
                                        batch,
                                        val_id,
                                        self.attack,
                                        self.device,
                                        self.number_of_steps,
                                        self.data_queue[:-1] + "_val/"
                                    ).start()
                                except StopIteration:
                                    val_iter = iter(self.val_data_set_loader)
                        else:
                            print("Data queue(Val) is full process is waiting...")
                            time.sleep(0.5)
                    else:
                        self.train_element_id = 0



