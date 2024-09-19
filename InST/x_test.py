import sys

import torch



def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {device}")



# def main():    
#     print(f"sys.version: {sys.version}")


if __name__ == "__main__":
    main()
