"""Exercice 4 : comparaison sur un plateau vide, avec une graine MCTS fixe."""

import argparse
import csv
from pathlib import Path

from algorithms import minimax, alphabeta, mcts
from game import create_board


def benchmark(board, algorithm, **kwargs):
    result = algorithm(board, **kwargs)
    return {
        "algorithm": algorithm.__name__,
        "move": result["move"],
        "score": result["score"],
        "nodes_explored": result["nodes_explored"],
        "time": result["time"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Enregistrer les graphiques sans ouvrir de fenetre")
    args = parser.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    board = create_board()
    depths = [1, 2, 3, 4, 5]
    budgets = [100, 500, 1000, 2000]
    results = []
    print(f"{'Algorithm':<12} {'Configuration':<18} {'Move':>5} {'Score':>12} {'Nodes':>10} {'Time (s)':>10}", flush=True)

    def record(algorithm, configuration, **kwargs):
        result = benchmark(board, algorithm, **kwargs)
        result["configuration"] = configuration
        results.append(result)
        print(f"{result['algorithm']:<12} {configuration:<18} {result['move']:>5} "
              f"{result['score']:>12.4f} {result['nodes_explored']:>10} {result['time']:>10.6f}", flush=True)
        return result

    for depth in depths:
        mm = record(minimax, f"Depth {depth}", depth=depth)
        ab = record(alphabeta, f"Depth {depth}", depth=depth)
        assert mm["score"] == ab["score"], "Minimax et Alpha-Beta doivent avoir le meme score"
        assert ab["nodes_explored"] <= mm["nodes_explored"], "Alpha-Beta explore trop de noeuds"

    for budget in budgets:
        record(mcts, f"{budget} simulations", num_simulations=budget, seed=42)

    output = Path(__file__).resolve().parent / "benchmark_results"
    output.mkdir(exist_ok=True)
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    fig, ax = plt.subplots()
    for name, label in [("minimax", "Minimax"), ("alphabeta", "Alpha-Beta")]:
        rows = [r for r in results if r["algorithm"] == name]
        ax.plot(depths, [r["nodes_explored"] for r in rows], marker="o", label=label)
    ax.set(xlabel="Profondeur de recherche", ylabel="Nombre de nœuds explores",
           title="Minimax et Alpha-Beta", xticks=depths)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output / "comparaison_noeuds.png", dpi=150)

    fig, ax = plt.subplots()
    rows = [r for r in results if r["algorithm"] == "mcts"]
    ax.plot(budgets, [r["time"] for r in rows], marker="o", label="MCTS (seed=42)")
    ax.set(xlabel="Nombre de simulations", ylabel="Temps d'execution (secondes)", title="Temps de calcul de MCTS")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output / "temps_mcts.png", dpi=150)
    print(f"\nResultats enregistres dans : {output}")
    print("Colonnes : 0 a 6. Le score MCTS est une recompense moyenne, pas une evaluation Minimax.")
    if not args.no_show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
