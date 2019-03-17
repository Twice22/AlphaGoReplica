# See documentation of the game here:
# https://senseis.xmp.net/?SmartGameFormat
# and
# https://www.red-bean.com/sgf/

import config

_SGF_COLUMNS = 'abcdefghijklmnopqrstuvwxyz'

_SGF_TEMPLATE = '''(;GM[1]FF[4]CA[UTF-8]AP[pigo_sgfwriter]RU[{ruleset}]
SZ[{board_size}]KM[{komi}]PW[{white_name}]PB[{black_name}]RE[{result}]
{game_moves})'''

app_name = "pigo"


# TODO: maybe check if the coords are in the right order (or reverse)
def coords_to_sgf(action):
    """
    Args:
        action (int): action in (1, n_rows * n_cols + 1)
    Returns:
        (str): sgf coordinates of the action
    """
    r = action // config.n_rows
    return "{}{}".format(_SGF_COLUMNS[r],
                         _SGF_COLUMNS[action - r * config.n_rows])


def action_to_sgf(idx, action):
    color = 'B' if idx % 2 == 0 else 'W'  # Game of GO starts with the Black player
    c = coords_to_sgf(action)
    return ";{color}[{coords}]".format(color=color, coords=c)


def write_sgf(
        actions_history,
        result_string,
        ruleset="Chinese",
        komi=5.5,
        black_name=app_name + "_B",
        white_name=app_name + "_W"):
    """
    Args:
        actions_history (list[int]): list of action in [1, n_rows * n_cols + 1]
        result_string (str): final result. "B+W" means Black wins by resign. "B+3.5"
            means Black wins by 3.5
        ruleset (str): Which rules to use. Default to "Chinese"
        komi (float): komi added to balance the game because as Black play first, they
            have some advantages
        black_name (str): name of the black player
        white_name (str): name of the white player
    Returns:
        (str): SGF formatted string
    """
    assert config.n_rows == config.n_cols
    board_size = config.n_rows
    game_moves = ''.join([action_to_sgf(i, action) for i, action in enumerate(actions_history)])
    result = result_string

    return _SGF_TEMPLATE.format(**locals())
