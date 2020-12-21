from src.data_load import load_single
from src.runner import run_for_single


def main(case, output_path):
    img = load_single(case)
    run_for_single(img, output_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Find skeleton and angle in hallux images')
    parser.add_argument('--case', type=int, required=True,
                        help='the path to workspace')
    parser.add_argument('--output', metavar='path', required=False,
                        help='output path')

    args = parser.parse_args()

    main(args.case, args.output)
