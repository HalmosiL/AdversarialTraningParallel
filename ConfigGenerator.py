from Cityscapes import CitySegmentation
import sys
import json

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('You have to give a config file path...')

        try: 
            print("Create config folder...")
            os.mkdir("./Configs")
            print("Folder created successfuly...")
        except OSError as error: 
            print("Data cache alredy exist...")  

    print("Use config:" + str(sys.argv[1]) + "...")

    CONFIG = json.load(open(sys.argv[1]))
    NAMES = []

    train_data_set_len = CitySegmentation(root=CONFIG['DATA_PATH'], split="train").__len__()
    val_data_set_len = CitySegmentation(root=CONFIG['DATA_PATH'], split="val").__len__()

    for i in range(CONFIG['NUMBER_OF_EXECUTORS']):
        name = "./Configs/config_" + str(i + 1) + ".json"

        data = {
            'ID': i + 1,
            'MODEL_CACHE': CONFIG['MODEL_CACHE'],
            'GPU_MAX_MEMORY_IN_USED': CONFIG['GPU_MAX_MEMORY_IN_USED'][i],
            'QUEUE_SIZE_TRAIN': CONFIG['QUEUE_SIZE_TRAIN'],
            'QUEUE_SIZE_VAL': CONFIG['QUEUE_SIZE_VAL'],
            'DATA_QUEUE': CONFIG['DATA_QUEUE'],
            'DATA_PATH': CONFIG['DATA_PATH'],
            'BATCH_SIZE': CONFIG['BATCH_SIZE'],
            'DEVICE': CONFIG['DEVICE'][i],
            'NUMBER_OF_STEPS': CONFIG['NUMBER_OF_STEPS'],
            'DATA_SET_START_INDEX_TRAIN': 0,
            'DATA_SET_END_INDEX_TRAIN': 48,
            'DATA_SET_START_INDEX_VAL': 0,
            'DATA_SET_END_INDEX_VAL': 48,
            'Allow_TO_RUN': True
        }

        with open(name, "w") as fp:
            json.dump(data , fp) 

        NAMES.append(name)
