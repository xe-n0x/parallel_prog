import argparse
from random import randint


def arg_parser():
    '''Парсер аргументов командной строки'''
    parser = argparse.ArgumentParser(prog="Make matrix")
    parser.add_argument("--file", type=str, help="file for matrix", required=True)
    parser.add_argument("--N", type=int, help="Number of rows", required=True)
    parser.add_argument("--M", type=int, help="Number of cols", required=True)
    parser.add_argument("--mini", type=int, help="Min_val", required=True)
    parser.add_argument("--maxi", type=int, help="Max_val", required=True)
    return parser.parse_args()


def create_matrix(file: str, n: int, m: int, mini: int, maxi: int):
    '''Создание матрицы'''
    with open(file, "w") as f:
        f.write(f"{n}\t{m}\n")
        for i in range (0,n):
            for j in range (0,m):
                if j!= m-1:
                    f.write(f"{randint(mini,maxi)} ")
                else:
                    f.write(f"{randint(mini,maxi)}\n")

        

if __name__ == "__main__":
    args = arg_parser()
    if args.mini>args.maxi:
        print("Минимальное число должно быть больше максимального")
    if args.M != args.N:
        print("Матрица не квадратная")
    else:
        create_matrix(args.file, args.N, args.M, args.mini, args.maxi)
    