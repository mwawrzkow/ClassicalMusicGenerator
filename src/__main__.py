import argparse
import os
import sys 
import shutil
# Instantiate the parser
parser = argparse.ArgumentParser(description='GAN RNN for generating music')

parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--dataset', type=str, default=None, help='Path to the directory containing the midi files')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint to load')
parser.add_argument('--num_files', type=int, default=None, help='Number of files to load')
parser.add_argument('--seq_length', type=int, default=100, help='Sequence length')
parser.add_argument('--output', type=str, default='output', help='Path to the output directory')
parser.add_argument('--seed', type=str, default=None, help='Path to the seed sequence')
parser.add_argument('--length', type=int, default=100, help='Length of the generated sequence')
parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for sampling')
parser.add_argument('--gan', type=str, default='gan', help='Type of GAN to use: rnn, gan, wgan-gp, wgan-lgp')
parser.add_argument('--num_generations', type=int, default=1, help='Number of midi files to create')
parser.add_argument('--continue_training', type=bool, default=False, help='Continue training from the checkpoint')

def download_dataset(dataset):
    # import hugging face hub download if it's not installed install by using pip
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        os.system("pip install huggingface_hub")
        from huggingface_hub import hf_hub_download
    # download the dataset from huggingface hub
    # check if the dataset exists
    if os.path.exists(dataset):
        return "./midi_data/MIDIs"
    # download the dataset
    # download to midi_data
    # create the directory if it does not exist
    if not os.path.exists("midi_data"):
        os.mkdir("midi_data")
    hf_hub_download(repo_id="asigalov61/Annotated-MIDI-Dataset", 
                repo_type="dataset", 
                filename="Annotated-MIDI-Dataset-CC-BY-NC-SA.zip",
                local_dir="midi_data")
    # extract the dataset
    print("Extracting the dataset")
    shutil.unpack_archive("midi_data/Annotated-MIDI-Dataset-CC-BY-NC-SA.zip", "midi_data")
    # remove the zip file
    print("Removing the zip file")
    os.remove("midi_data/Annotated-MIDI-Dataset-CC-BY-NC-SA.zip")
    print("Dataset downloaded and extracted")
    return "./midi_data/MIDIs"

if __name__ == '__main__':
    args = parser.parse_args()
    # perform checking the arguments
    # check if the directory containing the midi files exists
    if args.dataset is None or not os.path.exists(args.dataset):
        print(f"Directory {args.dataset} does not exist")
        print("Will download the dataset from huggingface hub")
        args.dataset = download_dataset("midi_data")
        
    if args.num_files is None:
        args.num_files = len(os.listdir(args.dataset))
    # check if the output directory exists
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    if args.gan not in ['gan', 'rnn']:
        print(f"Invalid GAN type: {args.gan}")
        sys.exit(1)
    if args.gan == 'rnn':
        import core_rnn
        core_rnn.run(args)
    else:
        import core_gan
        core_gan.run(args)