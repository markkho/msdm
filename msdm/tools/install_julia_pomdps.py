
def install_julia_pomdps(julia_path="julia"):
    import julia
    julia.install(julia=julia_path)

    from julia import Pkg
    Pkg.add([
        "POMDPs", "POMDPSimulators", "POMDPPolicies",
        "POMDPModelTools", "Distributions", "QuickPOMDPs",
        "QMDP", "SARSOP", "ARDESPOT", "IncrementalPruning"
    ])

if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if len(args) not in (0, 1):
        print('''
        Usage: python install_julia_pomdps.py [julia_path]
        ''')
        sys.exit(1)

    kw = dict()
    if len(args) == 1:
        kw['julia_path'] = args[0]
    install_julia_pomdps(**kw)
