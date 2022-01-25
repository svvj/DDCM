import math


class HashMap(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = {}

    def key(self, point):
        width = self.width
        height = self.height
        return (
            int((math.floor(point[0] / width))),
            int((math.floor(point[1] / height)))
        )

    def getPointsFromKey(self, key):
        valueFromKey = self.grid.get(key, [])

        if key not in self.grid:
            self.grid[key] = []

        return valueFromKey

    def insert(self, point):
        k = self.key(point)
        list = self.getPointsFromKey(k)
        list.append(point)
        dict = {k: list}
        self.grid.update(dict)
        return


    def delete(self, point):
        k = self.key(point)
        list = self.getPointsFromKey(k)
        if point in list:
            list.remove(point)
            dict = {k: list}
            self.grid.update(dict)
        else:
            raise ValueError('Point is not in hash table')


if __name__ == "__main__":
    hashMap = HashMap(10, 10)
    hashMap.insert((20, 20))
    grid = hashMap.getPointsFromKey((2, 2))
    print(grid)
