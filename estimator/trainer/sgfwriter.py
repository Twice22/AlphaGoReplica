# See documentation of the game here:
# https://senseis.xmp.net/?SmartGameFormat
# and
# https://www.red-bean.com/sgf/

from config import *

_SGF_COLUMNS = 'abcdefghijklmnopqrstuvwxyz'

_SGF_TEMPLATE = '''(;GM[1]FF[4]CA[UTF-8]AP[pigo_sgfwriter]RU[{ruleset}]
SZ[{board_size}]KM[{komi}]PW[{white_name}]PB[{black_name}]RE[{result}]
{game_moves})'''

app_name = "pigopigu"


# aa ba ca da ea fa ga ha ia ja ka la ma na oa pa qa ra sa
# ab bb cb db eb fb gb hb ib jb kb lb mb nb ob pb qb rb sb
# ac bc cc dc ec fc gc hc ic jc kc lc mc nc oc pc qc rc sc
# ad bd cd dd ed fd gd hd id jd kd ld md nd od pd qd rd sd
# ae be ce de ee fe ge he ie je ke le me ne oe pe qe re se
# af bf cf df ef ff gf hf if jf kf lf mf nf of pf qf rf sf
# ag bg cg dg eg fg gg hg ig jg kg lg mg ng og pg qg rg sg
# ah bh ch dh eh fh gh hh ih jh kh lh mh nh oh ph qh rh sh
# ai bi ci di ei fi gi hi ii ji ki li mi ni oi pi qi ri si
# aj bj cj dj ej fj gj hj ij jj kj lj mj nj oj pj qj rj sj
# ak bk ck dk ek fk gk hk ik jk kk lk mk nk ok pk qk rk sk
# al bl cl dl el fl gl hl il jl kl ll ml nl ol pl ql rl sl
# am bm cm dm em fm gm hm im jm km lm mm nm om pm qm rm sm
# an bn cn dn en fn gn hn in jn kn ln mn nn on pn qn rn sn
# ao bo co do eo fo go ho io jo ko lo mo no oo po qo ro so
# ap bp cp dp ep fp gp hp ip jp kp lp mp np op pp qp rp sp
# aq bq cq dq eq fq gq hq iq jq kq lq mq nq oq pq qq rq sq
# ar br cr dr er fr gr hr ir jr kr lr mr nr or pr qr rr sr
# as bs cs ds es fs gs hs is js ks ls ms ns os ps qs rs ss
def coords_to_sgf(action):
    """
    Args:
        action (int): action in (1, n_rows * n_cols + 1)
    Returns:
        (str): sgf coordinates of the action
    """
    r = action // FLAGS.n_rows
    return "{}{}".format(_SGF_COLUMNS[action - r * FLAGS.n_rows],
                         _SGF_COLUMNS[r])


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
    assert FLAGS.n_rows == FLAGS.n_cols
    board_size = FLAGS.n_rows
    game_moves = ''.join([action_to_sgf(i, action) for i, action in enumerate(actions_history)])
    result = result_string

    return _SGF_TEMPLATE.format(**locals())
