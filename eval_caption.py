from pycocoevalcap.eval import COCOEvalCap
import os.path as osp
import json
import argparse
import pdb

parser = argparse.ArgumentParser(
    description='Evaluate the caption results',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--result_file', type=str, 
                    help='path to evaluation file')

args = parser.parse_args()

def eval():
	global args

	with open(args.result_file, 'rt') as f:
    	    results = json.load(f)
        assert len(results) > 0, 'Cannot load results'
        # create cocoEval object by taking coco and cocoRes
	cocoEval = COCOEvalCap(results)
	# evaluate results
	cocoEval.evaluate()
	# print output evaluation scores


if __name__ == '__main__':
    eval()
