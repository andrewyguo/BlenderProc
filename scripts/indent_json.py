import argparse
import json 


parser = argparse.ArgumentParser()
parser.add_argument('json', help="Path to the json file to indent")
parser.add_argument('--output', '-o', help="Output file. Will overwrite existing file if left blank. ")
args = parser.parse_args()

with open(args.json, "r") as readfile:
    data = json.load(readfile)

if args.output is None:
    args.output = args.json

with open(args.output, "w") as writefile:
    json.dump(data, writefile, indent=4, separators=(',', ': '))
