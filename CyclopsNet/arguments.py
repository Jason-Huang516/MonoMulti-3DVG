import argparse
import torch


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data',
                        help='path to datasets')
    parser.add_argument('--dataset', default='rope3d',
                        help='dataset')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=45, type=float,
                        help='Seed.')
    parser.add_argument('--margin', default=0.3, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--learning_rate', default=3e-4, type=float,
                        help='Learning rate for the optimizer')
    parser.add_argument('--grad_clip', default=2.0, type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--decay_factor', default=2e-3, type=float,
                        help='weight_decay')
    parser.add_argument('--num_epochs', default=50, type=int,
                        help='Number of training epochs')
    parser.add_argument('--warmup_epochs', default=10, type=int,
                        help='Number of warmup epochs')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size for training')
    parser.add_argument('--d_model', default=768, type=int,
                        help='Dimension of the model')
    parser.add_argument('--latent_dim', default=512, type=int,
                        help='Dimension of the latent space')
    parser.add_argument('--num_heads', default=8, type=int,
                        help='Number of heads in multi-head attention')
    parser.add_argument('--num_layers', default=2, type=int,
                        help='Number of layers in the model')
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='Dropout probability')
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu",
                    help='Device to use for computation (cuda or cpu)')
    
    return parser