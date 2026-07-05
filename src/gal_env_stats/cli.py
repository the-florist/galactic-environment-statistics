"""
    Filename: cli.py
    Author: Ericka Florio
    Description: Command-line interface. Exposes the three analysis programs as
                 named subcommands and applies any --config / --cosmology
                 overrides before the parameters are loaded.
"""

import argparse

import gal_env_stats.config as config


def main(argv=None):
    """
        Parse the command line and run the requested analysis program.
    """
    parser = argparse.ArgumentParser(
        prog="gal-env-stats",
        description="Calculate and visualise the double distribution and the "
                    "most probable galactic outer density profile.")
    parser.add_argument(
        "--config", metavar="PATH",
        help="run-settings YAML to use (default: configs/run.yaml)")
    parser.add_argument(
        "--cosmology", metavar="NAME",
        help="cosmology preset to load, overriding the config's selector "
             "(e.g. concordance, eds)")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("growth-factor",
                   help="compute and plot the growth factor D(a)")
    sub.add_parser("density-profile",
                   help="compute and plot the most probable density profile")
    sub.add_parser("double-distribution",
                   help="compute and plot the double distribution")

    args = parser.parse_args(argv)

    # Apply config overrides BEFORE importing any module that reads parameters.
    if args.config:
        config.run_path_override = args.config
    if args.cosmology:
        config.cosmology_override = args.cosmology

    if args.command == "growth-factor":
        print("Visualising growth factor.")
        from gal_env_stats.analyses import growth_factor
        growth_factor.run()

    elif args.command == "density-profile":
        print("Visualising density profile.")
        from gal_env_stats.analyses import density_profile
        density_profile.run()

    elif args.command == "double-distribution":
        print("Visualising double distribution.")
        from gal_env_stats.analyses.double_distribution import DoubleDistribution
        DoubleDistribution().run()

    print("Program ended.")
