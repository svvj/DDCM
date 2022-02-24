import numpy as np
import cv2 as cv
import json

with open('m_array.json') as f:
    m_array = json.load(f)['m_array']

img = np.zeros((840, 1188, 3), np.uint8)
cv.rectangle(img, (0, 0), (1187, 839), (255, 255, 255), -1)     # White background

center = (594, 420)
padding = (80, 100)
pos = padding
diff = (int((1188 - padding[0]*2)/11), int((840 - padding[1]*2)/7))
black = (0, 0, 0)
for r in range(8):
    for c in range(12):
        if m_array[r][c] == 1:
            cv.circle(img, pos, 4, black, -1)
        elif m_array[r][c] == 2:
            cv.circle(img, (pos[0]-8, pos[1]), 4, black, -1)
            cv.circle(img, (pos[0]+8, pos[1]), 4, black, -1)
        elif m_array[r][c] == 3:
            cv.circle(img, (pos[0]-8, pos[1]), 4, black, -1)
            cv.circle(img, (pos[0]+8, pos[1]+8), 4, black, -1)
            cv.circle(img, (pos[0]+8, pos[1]-8), 4, black, -1)
        else:
            cv.circle(img, (pos[0] - 8, pos[1]), 4, black, -1)
            cv.circle(img, (pos[0] + 8, pos[1]), 4, black, -1)
            cv.circle(img, (pos[0], pos[1] - 8), 4, black, -1)
            cv.circle(img, (pos[0], pos[1] + 8), 4, black, -1)
        pos = (pos[0] + diff[0], pos[1])
    pos = (padding[0], pos[1] + diff[1])

cv.imshow('image', img)
cv.imwrite('marker.png', img)
cv.waitKey(0)
cv.destroyAllWindows()
