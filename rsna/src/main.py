import argparse

from pipeline.train import train_kfold
from pipeline.load_data import load_trn_data, load_test_data, load_trn_metadata, load_test_metadata


def main():
	pass


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument('-e', '--epochs', default='10', type=int)

	parser.parse_args()

	main()