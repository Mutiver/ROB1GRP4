import random


class TicTacToeBrain:

    turn = 0

    def flip_board(self, board):
        flipped_board = [list(row) for row in zip(*board)]
        return flipped_board

    def __init__(self):
        # You can initialize any required variables or data structures here.
        pass

    def best_move(self, board):
        # board = FlipBoard(board)
        self.turn = self.turn + 1
        if self.turn == 1:
            return [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1]
            ]
        if self.turn == 2:
            if board == [
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 1]
            ]:
                return [
                    [1, 0, 0],
                    [0, 2, 0],
                    [0, 0, 1]
                ]

        if self.turn == 3:
            if board == [
                    [1, 0, 2],
                    [0, 2, 0],
                    [0, 0, 1]
            ]:
                return [
                    [1, 0, 2],
                    [0, 2, 0],
                    [1, 0, 1]
                ]
        if self.turn == 4:
            if board == [
                    [1, 0, 2],
                    [0, 2, 0],
                    [1, 2, 1]
            ]:
                return [
                    [1, 0, 2],
                    [1, 2, 0],
                    [1, 2, 1]
                ]


# Example usage:
# You can create an instance of TicTacToeBrain and use the BestMove method.
# Replace this example with your actual game logic.

# Example board:
board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]

tic_tac_toe_brain = TicTacToeBrain()

for array in board:
    print(array)

print("\n")

board = tic_tac_toe_brain.BestMove(board)
for array in board:
    print(array)
    board = [
        [0, 0, 0],
        [0, 2, 0],
        [0, 0, 1]
    ]
print("\n")

board = tic_tac_toe_brain.BestMove(board)
for array in board:
    print(array)
    board = [
        [1, 0, 2],
        [0, 2, 0],
        [0, 0, 1]
    ]
print("\n")

board = tic_tac_toe_brain.BestMove(board)
for array in board:
    print(array)
    board = [
        [1, 0, 2],
        [0, 2, 0],
        [1, 2, 1]
    ]
print("\n")

board = tic_tac_toe_brain.BestMove(board)
for array in board:
    print(array)
