
def install(julia="julia"):
    import julia
    julia.install(julia=julia)

    from julia import Pkg
    Pkg.add([
        "POMDPs", "POMDPSimulators", "POMDPPolicies", "POMDPModelTools", "Distributions",
        "QMDP", "SARSOP", "ARDESPOT", "IncrementalPruning",
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
        kw['julia'] = args[0]
    install(**kw)