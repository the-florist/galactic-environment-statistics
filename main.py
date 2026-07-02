"""
    Read the program mode from the command line, and execute the corresponding
    module from the gal_env_stats package.
"""

import sys

if __name__ == '__main__':
    mode = sys.argv[1]
    match mode:
        case "1":
            print("Visualising growth factor.")
            import gal_env_stats.growth_factor as grw
            grw.run()

        case "2":
            print("Visualising density profile.")
            import gal_env_stats.density_profile as dpr 
            dpr.run()

        case "3":
            print("Visualising double distribution.")
            from gal_env_stats.double_distribution import DoubleDistribution
            dd = DoubleDistribution()
            dd.run()
            
        case _:
            print(f"Mode {mode} not supported.")
            exit()

    print("Program ended.")