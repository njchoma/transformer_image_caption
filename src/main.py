import os
import argparse

import torch

from models.caption_model import Caption_Model

#############################
#           DEVICE          #
#############################
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#####################################
#               UTILS               #
#####################################
def read_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument

    add_arg('--train_dir', type=str, required=True,
            help='Folder where training data is stored')
    add_arg('--artifacts_dir', type=str, required=True,
            help='Folder where all training artifacts are saved')
    add_arg('--name', type=str, required=True,
            help='Name of current training scheme')
    add_arg('--run_nb', type=int, default=0,
            help='Run number of current iteration of model')

    return parser.parse_args()


#####################################
#               MAIN                #
#####################################
def main():
    args = read_args()
    args.experiment_dir = os.path.join(args.artifacts_dir,
                                       args.name,
                                       str(args.run_nb))
    os.makedirs(args.experiment_dir, exist_ok=True)

    model = Caption_Model(dict_size=10, image_feature_dim=100)


if __name__ == "__main__":
    main()
