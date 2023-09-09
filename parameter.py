import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    parser.add_argument('--imsize', type=int, default=299)
    parser.add_argument('--total_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--load_model', type=str2bool, default=True)
    parser.add_argument('--shuffle', type=str2bool, default=True)
    parser.add_argument('--dataset', type=str, default='test_data', choices=['train_data', 'test_data'])
    parser.add_argument('--root_path', type=str, default='./data/output')
    parser.add_argument('--org_image_path', type=str, default='./data/train_data/org')
    parser.add_argument('--adv_image_path', type=str, default='./data/train_data/adv')
    parser.add_argument('--test_image_path', type=str, default='./data/test_data/adv')
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--model_save_path', type=str, default='models')

    return parser.parse_args()