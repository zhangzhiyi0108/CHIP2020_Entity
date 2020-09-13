import dynamic_yaml

from pathlib import Path

if __name__ == '__main__':
    config_file = '../base/config/config.yml'
    home_dir = str(Path.home())

    with open(config_file, mode='r', encoding='UTF-8') as f:
        config = dynamic_yaml.load(f)

    print(config.module.dir.saved)

    DEVICE = config.DEVICE

    train_file = config.data.conll2003.train
    # with open(file=train_file, mode='r', encoding="UTF-8") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         print(line)

    # Path(config.module.dir.saved).mkdir(parents=True, exist_ok=True)
