import threading
from Adversarial import model_immer_attack_auto_loss
import torch

class Gen(threading.Thread):
    number_of_gens = 0

    def __init__(self, model, batch, id_, attack, device, number_of_steps, data_queue):
        Gen.number_of_gens += 1

        threading.Thread.__init__(self)
        self.model = model
        self.batch = batch
        self.id_ = id_
        self.attack = attack
        self.device = device
        self.number_of_steps = number_of_steps
        self.data_queue = data_queue
 
    def run(self):
        print("Gen_", self.id_, " started..")
        image = self.batch[0].to(self.device)
        label = self.batch[1].to(self.device)

        image = model_immer_attack_auto_loss(
            image=image,
            model=self.model,
            attack=self.attack,
            number_of_steps=self.number_of_steps,
            device=self.device
        )

        Gen.number_of_gens -= 1

        torch.save(image, self.data_queue + 'image' + str(self.id_) + '.pt')
        torch.save(label, self.data_queue + 'label' + str(self.id_) + '.pt')
 