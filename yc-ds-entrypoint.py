import argparse
from autoencoder_learning_l1_l2 import learn_l1_l2


parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True, help='train dataset path')
parser.add_argument('--test', required=True, help='test dataset path')
parser.add_argument('--checkpoint', required=True, help='model checkpoint file path')
parser.add_argument('--output', help='Output file for model weights')


def main(train_df_fp: str, test_df_fp: str, checkpoint_fp: str, output_fp: str):
    learn_l1_l2(
        train_df_fp,
        test_df_fp,
        checkpoint_fp,
        output_fp,
    )


if __name__ == '__main__':
    args = parser.parse_args()
    main(
        train_df_fp=args.train,
        test_df_fp=args.test,
        checkpoint_fp=args.checkpoint,
        output_fp=args.output,
    )
