import random
from collections.abc import Callable


def print_board(board: list[str]) -> None:
    """Display the current board with position numbers on empty cells."""
    print()
    for i in range(0, 9, 3):
        row = []
        for j in range(3):
            cell = board[i + j]
            row.append(cell if cell != " " else str(i + j + 1))
        print(f" {row[0]} | {row[1]} | {row[2]} ")
        if i < 6:
            print("---|---|---")
    print()


def check_win(board: list[str], player: str) -> bool:
    """Return True if the given player has won."""
    win_patterns = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    ]
    for pattern in win_patterns:
        if all(board[i] == player for i in pattern):
            return True
    return False


def is_full(board: list[str]) -> bool:
    """Return True if the board has no empty cells left."""
    return all(cell != " " for cell in board)


def minimax(board: list[str], depth: int, is_maximizing: bool,
            ai: str, human: str) -> int:
    """Minimax. Returns a score for the board state."""
    if check_win(board, ai):
        return 10 - depth
    if check_win(board, human):
        return depth - 10
    if is_full(board):
        return 0

    if is_maximizing:
        best = -10
        for i in range(9):
            if board[i] == " ":
                board[i] = ai
                score = minimax(board, depth + 1, False, ai, human)
                board[i] = " "
                best = max(best, score)
        return best

    best = 10
    for i in range(9):
        if board[i] == " ":
            board[i] = human
            score = minimax(board, depth + 1, True, ai, human)
            board[i] = " "
            best = min(best, score)
    return best


def ai_move(board: list[str], ai: str, human: str) -> int:
    """Return the best move for the AI using minimax (hard mode)."""
    best_score = -10
    best_pos = 0
    for i in range(9):
        if board[i] == " ":
            board[i] = ai
            score = minimax(board, 0, False, ai, human)
            board[i] = " "
            if score > best_score:
                best_score = score
                best_pos = i
    return best_pos


def ai_move_easy(board: list[str]) -> int:
    """Return a random legal move (easy mode)."""
    empty = [i for i in range(9) if board[i] == " "]
    return random.choice(empty)


def get_human_move(board: list[str], player: str) -> int:
    """Prompt the human for a valid move and return the board index."""
    while True:
        try:
            move = int(input(f"Player {player}, enter position (1-9): ")) - 1
        except ValueError:
            print("Invalid input. Enter a number 1-9.")
            continue
        if 0 <= move <= 8 and board[move] == " ":
            return move
        print("That spot is taken or out of range. Try again.")


MoveFn = Callable[[list[str], str], int]


def run_game(get_move_o: MoveFn, get_move_x: MoveFn, players: tuple[str, str]) -> None:
    """
    Generic game loop.

    get_move_x and get_move_o are callables that take (board, player) and
    return an index. players is a tuple ("X", "O") defining turn order.
    """
    board = [" "] * 9
    current = players[0]
    get_move = {players[0]: get_move_x, players[1]: get_move_o}

    while True:
        print_board(board)
        move = get_move[current](board, current)
        board[move] = current
        if check_win(board, current):
            print_board(board)
            print(f"Player {current} wins!")
            return
        if is_full(board):
            print_board(board)
            print("It's a tie!")
            return
        current = players[1] if current == players[0] else players[0]


def play_vs_human() -> None:
    """Start a two-player game."""
    run_game(get_human_move, get_human_move, ("X", "O"))


def play_vs_ai() -> None:
    """Play against the AI with configurable first move and difficulty."""
    first = input("Who goes first? (y)ou or (a)i: ").strip().lower()
    difficulty = input("AI difficulty (e)asy or (h)ard: ").strip().lower()
    is_hard = difficulty == "h"

    if first == "a":
        def ai_move_for_x(board: list[str], player: str) -> int:
            return ai_move(board, player, "O") if is_hard else ai_move_easy(board)

        def human_move_for_o(board: list[str], player: str) -> int:
            return get_human_move(board, player)

        print("\nAI is X. You are O.\n")
        run_game(human_move_for_o, ai_move_for_x, ("X", "O"))
    else:
        def ai_move_for_o(board: list[str], player: str) -> int:
            return ai_move(board, player, "X") if is_hard else ai_move_easy(board)

        def human_move_for_x(board: list[str], player: str) -> int:
            return get_human_move(board, player)

        print("\nYou are X. AI is O.\n")
        run_game(ai_move_for_o, human_move_for_x, ("X", "O"))


def main() -> None:
    """Entry point — menu and replay loop."""
    print("Tic Tac Toe")
    print("Positions are 1-9, left to right, top to bottom.\n")
    while True:
        choice = input("Play against (h)uman or (a)i? ").strip().lower()
        if choice == "a":
            play_vs_ai()
        else:
            play_vs_human()
        again = input("\nPlay again? (y/n): ").strip().lower()
        if again != "y":
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
