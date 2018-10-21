import re


def pre_engine(s):
    s = re.sub("[^\t\n -~]", "", s)
    s = s.split("#")[0]
    s = s.replace("\t", " ")
    return s


def pre_controller(s):
    s = re.sub("[^\t\n -~]", "", s)
    s = s.replace("\t", " ")
    return s


def gtp_boolean(b):
    return "true" if b else "false"


def gtp_list(l):
    return "\n".join(l)


def gtp_color(color):
    # an arbitrary choice amongst a number of possibilities
    return {BLACK: "B", WHITE: "W"}[color]


# TODO: remove gtp_vertex
def coord_to_gtp(action, board_size):
    """ From 1D coord (0 for position (0,0)) to J1 """

    # the action 'board_size ** 2' is the 'pass' action
    # action 'board_size ** 2 + 1' is the 'ressign' action
    if action == board_size ** 2:
        return "pass"
    elif action == board_size ** 2 + 1:
        return 'resign'
    return "{}{}".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[coord % board_size],\
        board_size - coord // board_size)


def gtp_coord(gtp_action, board_size):
    """ From gtp coord to pachi coord """
    if gtp_coord == "pass":
        return board_size ** 2
    elif gtp_coord == "resign":
        return board_size ** 2 + 1

    assert len(gtp_action) == 2
    letter, digit = list(gtp_action)
    x = "ABCDEFGHJKLMNOPQRSTYVWYZ".index(letter) + 1
    y = board_size - int(digit)
    return y * board_size + x - 1

def gtp_move(color, vertex):
    return " ".join([gtp_color(color), gtp_vertex(vertex)])


def parse_message(message):
    message = pre_engine(message).strip()
    first, rest = (message.split(" ", 1) + [None])[:2]
    if first.isdigit():
        message_id = int(first)
        if rest is not None:
            command, arguments = (rest.split(" ", 1) + [None])[:2]
        else:
            command, arguments = None, None
    else:
        message_id = None
        command, arguments = first, rest

    return message_id, command, arguments


WHITE = -1
BLACK = +1
EMPTY = 0

PASS = (0, 0) # TODO: delete this with vertex_in_range?
RESIGN = "resign"


def parse_color(color):
    if color.lower() in ["b", "black"]:
        return BLACK
    elif color.lower() in ["w", "white"]:
        return WHITE
    else:
        return False


MIN_BOARD_SIZE = 7
MAX_BOARD_SIZE = 19


def format_success(message_id, response=None):
    if response is None:
        response = ""
    else:
        response = " {}".format(response)
    if message_id:
        return "={}{}\n\n".format(message_id, response)
    else:
        return "={}\n\n".format(response)


def format_error(message_id, response):
    if response:
        response = " {}".format(response)
    if message_id:
        return "?{}{}\n\n".format(message_id, response)
    else:
        return "?{}\n\n".format(response)


class Engine(object):

    def __init__(self, game_obj, komi=6.5, board_size=9, name="gtp (python library)", version="0.2"):

        self.board_size = board_size
        self.komi = komi

        self._game = game_obj
        self._game.reset()

        self._name = name
        self._version = version

        self.disconnect = False

        self.known_commands = [
            field[4:] for field in dir(self) if field.startswith("cmd_")]

    def send(self, message):
        message_id, command, arguments = parse_message(message)
        if command in self.known_commands:
            try:
                return format_success(
                    message_id, getattr(self, "cmd_" + command)(arguments))
            except ValueError as exception: # TODO: do we need to delete this?
                return format_error(message_id, exception.args[0])
        else:
            return format_error(message_id, "unknown command")

    # TODO: can we delete this?
    def vertex_in_range(self, vertex):
        if vertex == PASS:
            return True
        if 1 <= vertex[0] <= self.board_size and 1 <= vertex[1] <= self.board_size:
            return True
        else:
            return False

    # commands

    def cmd_protocol_version(self, arguments):
        return 2

    def cmd_name(self, arguments):
        return self._name

    def cmd_version(self, arguments):
        return self._version

    def cmd_known_command(self, arguments):
        return gtp_boolean(arguments in self.known_commands)

    def cmd_list_commands(self, arguments):
        return gtp_list(self.known_commands)

    def cmd_quit(self, arguments):
        self.disconnect = True

    def cmd_boardsize(self, arguments):
        if arguments.isdigit():
            size = int(arguments)
            if MIN_BOARD_SIZE <= size <= MAX_BOARD_SIZE:
                self.board_size = size
            else:
                raise ValueError("unacceptable size")
        else:
            raise ValueError("non digit size")

    def cmd_clear_board(self, arguments):
        self._game.reset()

    def cmd_komi(self, arguments):
        try:
            komi = float(arguments)
            self.komi = komi
        except ValueError:
            raise ValueError("syntax error")

    def cmd_play(self, arguments):
        move = gtp_to_coord(arguments, self.board_size)
        if self._game.solo_play(move):
            return ""
        raise ValueError("illegal move")

    def cmd_genmove(self, arguments):
        c = parse_color(arguments)
        if c:
            move = self._game.get_move(c) # TODO: change for solo_play
            return coord_to_gtp(move, self.board_size)
        else:
            raise ValueError("unknown player: {}".format(arguments))
