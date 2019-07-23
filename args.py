import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--model',
		type=str,
		default='CLEVR',
		help='model to train')
	parser.add_argument(
		'--root',
		type=str,
		default='/mnt/nas2/GrimaRepo/datasets/CLEVR_v1.0/features',
		help='Dataset path',
		)
	parser.add_argument(
		'--comet',
		type=int,
		default=1,
		help='Log exp to comet',
		)
	parser.add_argument(
		'--exp_name',
		type=str,
		default='MAC',
		help='Experiment Name',
		)
	args = parser.parse_args()
	return args