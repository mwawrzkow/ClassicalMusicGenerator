import argparse
import os
import sys 
# Instantiate the parser
parser = argparse.ArgumentParser(description='GAN RNN for generating music')

parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--dataset', type=str, default='dataset', help='Path to the directory containing the midi files')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint to load')
parser.add_argument('--num_files', type=int, default=100, help='Number of files to load')
parser.add_argument('--seq_length', type=int, default=100, help='Sequence length')
parser.add_argument('--output', type=str, default='output', help='Path to the output directory')
parser.add_argument('--seed', type=str, default=None, help='Path to the seed sequence')
parser.add_argument('--length', type=int, default=100, help='Length of the generated sequence')
parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for sampling')
parser.add_argument('--gan', type=str, default='gan', help='Type of GAN to use: rnn, gan, wgan-gp, wgan-lgp')
parser.add_argument('--num_generations', type=int, default=1, help='Number of midi files to create')

if __name__ == '__main__':
    args = parser.parse_args()
    # perform checking the arguments
    # check if the directory containing the midi files exists
    if not os.path.exists(args.dataset):
        print(f"Directory {args.dataset} does not exist")
        sys.exit(1)
    # check if the output directory exists
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    if args.gan not in ['gan', 'wgan-gp', 'wgan-lgp', 'rnn']:
        print(f"Invalid GAN type: {args.gan}")
        sys.exit(1)
    if args.gan == 'rnn':
        import core_rnn
        core_rnn.run(args)
    else:
        import core_gan
        core_gan.run(args)