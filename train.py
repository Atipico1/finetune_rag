import argparse
from utils import str2bool


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, default="")
    parser.add_argument("--dataset", type=str, required=True, default="")