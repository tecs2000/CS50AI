"""
Tic Tac Toe Player
"""
from copy import deepcopy
from math import inf

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """

    if board == initial_state:
        return X
    if terminal(board):
        return 'End'

    count_X, count_O = 0, 0
    for row in board:
        count_X += row.count(X)
        count_O += row.count(O)
    return X if count_X == count_O else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    if terminal(board):
        return 'End'

    action = set()
    for row in range(3):
        for column in range(3):
            if board[row][column] is EMPTY:
                action.add((row, column))
    return action


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    board_copy = deepcopy(board)

    if board_copy[action[0]][action[1]] is not EMPTY or terminal(board):
        raise Exception('invalid action!')

    board_copy[action[0]][action[1]] = player(board)
    return board_copy


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    # winner in some line
    for row in board:
        if row.count(X) == 3:
            return X
        if row.count(O) == 3:
            return O

    # winner in some column
    for i in range(3):
        count_x, count_o = 0, 0
        for j in range(3):
            if board[j][i] == X:
                count_x += 1
            elif board[j][i] == O:
                count_o += 1
        if count_x == 3:
            return X
        if count_o == 3:
            return O

    # winner in main diagonal
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not EMPTY:
        return X if board[0][0] == X else O

    # winner in secondary diagonal
    if board[2][0] == board[1][1] == board[0][2] and board[2][0] is not EMPTY:
        return X if board[2][0] == X else O

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is None:
        for row in range(3):
            for column in range(3):
                if board[row][column] is EMPTY:
                    return False  # game isnt over
        return True  # if theres no empty cell, games ended in tie
    return True


def utility(depth, board):
    """
    Returns punctuation considering the necessary
    depth to find the terminal state.
    """
    # depth approach

    if winner(board) is None:
        return 0
    if winner(board) == player(board):
        return 10 - depth
    return -10 + depth

    # general approach

    # if winner(board) is None:
    #     return 0
    # if winner(board) == player(board):
    #     return 1
    # return -1


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    best = (0, 0)

    alpha = -inf
    beta = inf

    for move in actions(board):
        score = -negamax_abpruning(0, result(board, move), -beta, -alpha)
        if score >= beta:
            return best
        if score > alpha:
            alpha = score
            best = move
    return best


def negamax_abpruning(depth, board, alpha, beta):
    if terminal(board):
        return utility(depth, board)

    for move in actions(board):
        bestScore = -negamax_abpruning(depth + 1,
                                       result(board, move), -beta, -alpha)
        if bestScore >= beta:
            return beta
        if bestScore > alpha:
            alpha = bestScore
    return alpha

# def minimax(board):
#     for move in actions(board):
#         score = negamax(result(board, move))
#         if score > best[1]:
#             best = (move, score)
#     return best[0]

# def negamax(board):
#     if terminal(board):
#         return utility(board)

#     best = -inf
#     for move in actions(board):
#         score = -negamax(result(board, move))
#         if score > best:
#             best = score
#     return best
