"""
    Read the program mode from the command line, and execute the corresponding 
    module from src/
"""

import sys

if __name__ == '__main__':
    mode = sys.argv[1]
    match mode:
        case "1":
            print("Visualising growth factor.")
            import src.growth_factor as grw
            grw.run()
        case "2":
            print("Visualising density profile.")
            import src.density_profile as dpr 
            dpr.run()
        case "3":
            print("Visualising double distribution.")
            import src.double_distribution as dd 
            dd.run()
        case _:
            print(f"Mode {mode} not supported.")
            exit()

    print("Program ended.")