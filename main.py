import argparse
import yaml

from trainer import *
from utils.utils import *

# 设置随机种子
'''
seed_value = 42
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed_value)
'''

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run.")

    parser.add_argument("--config_path",
                        type=str,
                        default='config.yaml',
                        help="Config path of trainer.")

    return parser.parse_args()


def main():
    args = parameter_parser()
    config = load_config(args.config_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_num'])

    trainer = Trainer(config)
    print(f"Total parameters:{calculate_total_params(trainer.model)}")

    if config['load_model']:
        trainer.load(config['load_model'])
        # trainer.train()
        # trainer.load(config['model_name'] + '_state_best.pth')
    else:
        trainer.train()
        trainer.load(config['model_name'] + '_state_best.pth')
    
    # print(f"k = {trainer.model.k.sum().item()}, c = {trainer.model.c.sum().item()}, b = {trainer.model.b.sum().item()}")
    trainer.eval()


if __name__ == "__main__":
    main()