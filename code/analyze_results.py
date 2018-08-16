import argparse

import visualize as vz
import data_processing as dp


def main():
    parser = argparse.ArgumentParser(description='Simple analyzer for active learning result files.')
    parser.add_argument('-r', '--results', nargs='+', help='All result files to analyze')
    parser.add_argument('-u', '--upper_bound', type=float, default=1.0, help='Upper bound to be plotted in the graphs.')

    args = parser.parse_args()
    results = [dp.read_active_learning_history(single_file) for single_file in args.results]
    captions = [single_file.split('/')[-1] for single_file in args.results]

    vz.plot_several(results, captions, args.upper_bound)

if __name__ == '__main__':
    main()
