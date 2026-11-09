"""Tests des decisions de jeu et de l'integration Flask."""

import unittest

from algorithms import minimax, alphabeta, mcts
from app import app
from game import AI, HUMAN, WIN_SCORE, create_board, make_move


def position(columns):
    board = create_board()
    for turn, column in enumerate(columns):
        board = make_move(board, column, HUMAN if turn % 2 == 0 else AI)
    return board


class SearchTests(unittest.TestCase):
    def test_win_and_block(self):
        win = position([0, 6, 1, 6, 2, 6, 4])
        block = position([0, 6, 1, 6, 2])
        for algorithm in (minimax, alphabeta):
            self.assertEqual(algorithm(win, depth=2)["move"], 6)
            self.assertEqual(algorithm(block, depth=2)["move"], 3)

    def test_search_agreement_and_immutable_board(self):
        for board in (create_board(), position([3, 2, 3, 4, 1])):
            original = [row[:] for row in board]
            for depth in (1, 2, 3):
                mm = minimax(board, depth)
                ab = alphabeta(board, depth)
                self.assertEqual((mm["move"], mm["score"]), (ab["move"], ab["score"]))
                self.assertLessEqual(ab["nodes_explored"], mm["nodes_explored"])
            self.assertEqual(board, original)

    def test_terminal_board(self):
        board = position([0, 6, 1, 6, 2, 6, 4, 6])
        for algorithm in (minimax, alphabeta, mcts):
            result = algorithm(board)
            self.assertIsNone(result["move"])
            self.assertEqual(result["nodes_explored"], 0)
            if algorithm is not mcts:
                self.assertEqual(result["score"], WIN_SCORE)

    def test_mcts_visits_and_seed(self):
        board = position([3])
        original = [row[:] for row in board]
        first = mcts(board, num_simulations=100, seed=42)
        second = mcts(board, num_simulations=100, seed=42)
        self.assertEqual(first["move"], second["move"])
        self.assertEqual(first["search_tree"], second["search_tree"])
        self.assertEqual(board, original)
        nodes = first["search_tree"]["nodes"]
        self.assertEqual(nodes[0]["visits"], 100)
        children = [node for node in nodes if node["parent"] == 0]
        self.assertEqual(sum(node["visits"] for node in children), 100)
        self.assertEqual(first["move"], max(children, key=lambda node: node["visits"])["move"])
        self.assertTrue(all(0 <= node["mean_reward"] <= 1 for node in nodes))

    def test_human_then_ai_through_api(self):
        app.config["TESTING"] = True
        with app.test_client() as client:
            self.assertEqual(client.get("/").status_code, 200)
            for algorithm in ("random", "minimax", "alphabeta", "mcts"):
                client.post("/api/new-game")
                self.assertEqual(client.post("/api/play", json={"column": 3}).status_code, 200)
                response = client.post("/api/ai-move", json={
                    "algorithm": algorithm, "depth": 2, "num_simulations": 100,
                })
                self.assertEqual(response.status_code, 200, response.get_json())
                state = response.get_json()["state"]
                self.assertEqual(state["turn"], HUMAN)
                self.assertEqual(sum(cell == AI for row in state["board"] for cell in row), 1)


if __name__ == "__main__":
    unittest.main()
