import math


def setDefault(dict, key):
    valueFromKey = dict.get(key, [])

    if key not in dict:
        dict[key] = []

    return valueFromKey


class HashMap(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = {}

    def key(self, point):
        width = self.width
        height = self.height
        return (
            int((math.floor(point[0] / width)) * width),
            int((math.floor(point[1] / height)) * height)
        )

    def insert(self, point):
        k = self.key(point)
        setDefault(self.grid, k).append(point)

    def getValuesFromKey(self, point):
        return setDefault(self.grid, self.key(point))
