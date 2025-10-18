import sys

if __name__ == '__main__':
    mode = sys.argv[1]
    match mode:
        case "1":
            # run growth-factor
            pass
        case "2":
            # run density-profile
            pass
        case "3":
            # run double-distribution
            pass
        case _:
            print(f"Mode {mode} not supported.")
            exit()