"""Cost-based search algorithms for the Pokémon environment."""

from __future__ import annotations

import heapq
import random
from itertools import count

from pokemon_game import (
    GameMap,
    State,
    movement_cost,
    successors,
)


def reconstruct_path(
    parents: dict[State, State | None],
    goal: State,
) -> list[State]:
    """Reconstruct a path from a parent dictionary."""
    path: list[State] = []
    current: State | None = goal

    while current is not None:
        path.append(current)
        current = parents[current]

    path.reverse()
    return path


def path_cost(
    game_map: GameMap,
    path: list[State],
) -> int:
    """Return the total cost of a path."""
    return sum(
        movement_cost(game_map, state)
        for state in path[1:]
    )


def manhattan_distance(
    state: State,
    goal: State,
) -> int:
    """Return the Manhattan distance between two states."""
    return (
        abs(state[0] - goal[0])
        + abs(state[1] - goal[1])
    )


def generate_random_path(
    game_map: GameMap,
    start: State,
    goal: State,
    seed: int | None = None,
) -> list[State]:
    """Generate a random valid path from start to goal.

    This function is provided only to demonstrate the visualization.
    It is not guaranteed to return a minimum-cost or shortest path.

    The function performs a randomized depth-first exploration: the valid
    successor states are shuffled before being added to the frontier.

    Parameters
    ----------
    game_map : GameMap
        Weighted Pokémon map.
    start : State
        Initial trainer position.
    goal : State
        Pokémon Center position.
    seed : int or None, default=None
        Random seed used to make the result reproducible.

    Returns
    -------
    list[State]
        A valid path beginning at ``start`` and ending at ``goal``.

    Raises
    ------
    RuntimeError
        If no path exists between the start and the goal.
    """
    rng = random.Random(seed)

    frontier = [start]

    parents: dict[State, State | None] = {
        start: None,
    }

    while frontier:
        current = frontier.pop()

        if current == goal:
            return reconstruct_path(parents, goal)

        next_states = successors(game_map, current)
        rng.shuffle(next_states)

        for next_state in next_states:
            if next_state in parents:
                continue

            parents[next_state] = current
            frontier.append(next_state)

    raise RuntimeError(
        "No path exists between the start and the goal."
    )

def dijkstra(
    game_map: GameMap,
    start: State,
    goal: State,
) -> tuple[list[State], int]:
    """Return a minimum-cost path using Dijkstra's algorithm. """

    frontiere = []
    compteur = count()

    heapq.heappush(frontiere, (0, next(compteur), start))

    couts = {start: 0}
    parents = {start: None}
    etats_explores = 0

    while frontiere:
        cout, _, actuel = heapq.heappop(frontiere)

        # on ignore les anciens chemins plus coûteux
        if cout != couts[actuel]:
            continue

        etats_explores += 1

        # Vérifier si on atteint l'objectif (state)
        if actuel == goal:
            chemin = reconstruct_path(parents, goal)
            return chemin, etats_explores

        # on explore les cases voisines
        for voisin in successors(game_map, actuel):
            nouveau_cout = cout + movement_cost(game_map, voisin)

            if voisin not in couts or nouveau_cout < couts[voisin]:
                couts[voisin] = nouveau_cout
                parents[voisin] = actuel

                heapq.heappush(
                    frontiere,
                    (nouveau_cout, next(compteur), voisin)
                )

    raise RuntimeError("On a trouvé aucun chemin")

    ...


def a_star(
    game_map: GameMap,
    start: State,
    goal: State,
) -> tuple[list[State], int]:
    """Return a minimum-cost path using A* search. """

    frontiere = []
    compteur = count()

    heuristique = manhattan_distance(start, goal)

    heapq.heappush(frontiere, (heuristique, next(compteur), 0, start))

    couts = {start: 0}
    parents = {start: None}
    etats_explores = 0

    while frontiere:
        priorite, _, cout, actuel = heapq.heappop(frontiere)

        # On ignore les chemins obsolètes
        if cout != couts[actuel]:
            continue

        etats_explores += 1

        # Vérifier si on attenint l'objectif (state)
        if actuel == goal:
            chemin = reconstruct_path(parents, goal)
            return chemin, etats_explores

        # on explore les voisins
        for voisin in successors(game_map, actuel):
            nouveau_cout = cout + movement_cost(game_map, voisin)

            if voisin not in couts or nouveau_cout < couts[voisin]:
                couts[voisin] = nouveau_cout
                parents[voisin] = actuel

                # Calcul de f(s) = g(s) + h(s)
                heuristique = manhattan_distance(voisin, goal)
                nouvelle_priorite = nouveau_cout + heuristique

                heapq.heappush(
                    frontiere,
                    (
                        nouvelle_priorite,
                        next(compteur),
                        nouveau_cout,
                        voisin
                    )
                )

    raise RuntimeError("On a trouvé aucun chemin")
