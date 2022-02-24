from itertools import product

import numpy as np
import random

def marker_generator(r, c):
    marker_set = [1, 2, 3, 4]
    format = list(product(marker_set, marker_set, marker_set, marker_set))
    used_format = (np.ones(256, dtype=int) * -1).tolist()
    board = (np.ones((8, 12), dtype=int) * -1).tolist()
    r = 0
    while r < 7:
        c = 0
        while c < 11:
            square = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
            if -1 not in square:
                c += 1
                continue
            filled_pos = [i for i, value in enumerate(square) if value != -1]

            possible_markers = []
            for fmt in format:
                check = [0] * len(filled_pos)
                for i, f in enumerate(filled_pos):
                    if fmt[f] == square[f]:
                        check[i] = 1
                if sum(check) == len(filled_pos):
                    possible_markers.append(fmt)

            cnt = 0
            find_marker_rc = False
            while cnt <= len(possible_markers):
                marker_rc = possible_markers[random.randrange(len(possible_markers))]
                if used_format[format.index(marker_rc)] == -1:
                    used_format[format.index(marker_rc)] = 1
                    find_marker_rc = True
                    break
                else:
                    cnt += 1

            if find_marker_rc:
                board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1] = marker_rc
                print(r, c, ':', board[r][c], board[r][c + 1], board[r + 1][c], board[r + 1][c + 1])
                c += 1
            else:
                if c == 0:
                    r -= 1
                    c = 11
                else:
                    c -= 1
                print(r, c, ': Duplicate!', board[r][c], board[r][c + 1], board[r + 1][c], board[r + 1][c + 1])
                used_format[format.index((board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]))] = -1
                board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1] = -1, -1, -1, -1
        r += 1

    return board


if __name__ == "__main__":
    board = marker_generator(8, 12)
    l = []
    s = True
    for r in range(7):
        for c in range(11):
            square = (board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1])
            if square in l:
                s = False
                break
            l.append(square)
        if not s:
            break
    if s:
        print('perfect')
    else:
        print('wrong')

