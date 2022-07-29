from ConfigGenerator import config_generator
import subprocess
import glob
import json

def start_all():
    configs = glob.glob("./Configs/*.json")

    for config in configs:
        log_path = json.load(open(config))["LOG_PATH"]
        bashCommand = ["./start_one.sh", config, log_path]
        list_files = subprocess.Popen(bashCommand, stdout=subprocess.PIPE)

if __name__ == '__main__':
    config_generator()
    start_all()
