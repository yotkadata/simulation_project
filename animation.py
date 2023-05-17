import cv2
import numpy as np
import time

TILE_SIZE = 32

FLOOR = """
####################
####################
#BBBBDDDDDSSSSSFFFF#
#BBBBDDDDDSSSSSFFFF#
#BBB##DDD##SSS##FFF#
#BBB##DDD##SSS##FFF#
#BBB##DDD##SSS##FFF#
#BBB##DDD##SSS##FFF#
#BBB##DDD##SSS##FFF#
#BBB##DDD##SSS##FFF#
#BBB##DDD##SSS##FFF#
#BBB##DDD##SSS##FFF#
#BBB##DDD##SSS##FFF#
BBBBBDDDDDSSSSEEEEEE
BBBBBDDDDDSSSSEEEEEE
##CC##CC##CC##EEEEEE
##CC##CC##CC##EEEEEE
##CC##CC##CC##EEEEEE
CCCCCCCCCCCCCCEEEEEE
CCCCCCCCCCCCCCEEEEEE
""".strip()


class SupermarketMap:
    """
    Visualizes the supermarket background.
    """

    def __init__(self, floor, tiles):
        """
        floor: a string with each character representing a tile
        tiles: a numpy array containing all the tile images
        """
        self.tiles = tiles
        # split the floor string into a two dimensional matrix
        self.contents = [list(row) for row in floor.split("\n")]
        self.ncols = len(self.contents[0])
        self.nrows = len(self.contents)
        self.image = np.zeros(
            (self.nrows * TILE_SIZE, self.ncols * TILE_SIZE, 3), dtype=np.uint8
        )
        self.prepare_map()

    def extract_tile(self, row, col):
        """extract a tile array from the tiles image"""
        y = row * TILE_SIZE
        x = col * TILE_SIZE
        return self.tiles[y : y + TILE_SIZE, x : x + TILE_SIZE]

    def get_tile(self, char):
        """returns the array for a given tile character"""
        if char == "#":  # Wall
            return self.extract_tile(0, 0)
        elif char == "E":  # Entrance
            return self.extract_tile(0, 1)
        elif char == "F":  # Fruits
            return self.extract_tile(0, 2)
        elif char == "S":  # Spices
            return self.extract_tile(0, 3)
        elif char == "D":  # Dairy
            return self.extract_tile(0, 4)
        elif char == "B":  # Drinks
            return self.extract_tile(1, 0)
        elif char == "C":  # Checkout
            return self.extract_tile(1, 1)
        else:
            return self.extract_tile(1, 2)

    def prepare_map(self):
        """prepares the entire image as a big numpy array"""

        # Get all unique characters in floor plan
        chars = "".join(set(FLOOR)).replace("\n", "")

        # Create empty dict to store possible positions
        self.positions = {key: [] for key in chars}

        for row, line in enumerate(self.contents):
            for col, char in enumerate(line):
                # Save all possible positions of letters in a dict
                self.positions[char].append((row, col))

                bm = self.get_tile(char)
                y = row * TILE_SIZE
                x = col * TILE_SIZE
                self.image[y : y + TILE_SIZE, x : x + TILE_SIZE] = bm

    def draw(self, frame):
        """
        draws the image into a frame
        """
        frame[0 : self.image.shape[0], 0 : self.image.shape[1]] = self.image

    def write_image(self, filename):
        """writes the image into a file"""
        cv2.imwrite(filename, self.image)


class Customer:
    def __init__(self, supermarket, tiles):
        """
        Customer object
        supermarket: A SuperMarketMap object
        avatar : a numpy array containing a 32x32 tile image
        """

        self.supermarket = supermarket
        self.row, self.col = self.get_rand_position("E")
        self.tiles = tiles
        self.avatar = self.extract_tile(7, 0)
        self.section = "C"

    def __repr__(self) -> str:
        return f"Customer object"

    def get_rand_position(self, section=None):
        """
        Randomly select a position in a given section.
        """
        if section == None:
            section = self.section

        choices = self.supermarket.positions[section]
        i = np.random.choice(len(choices))

        return choices[i]

    def draw(self, frame):
        row1 = self.row * TILE_SIZE
        row2 = row1 + self.avatar.shape[0]

        col1 = self.col * TILE_SIZE
        col2 = col1 + self.avatar.shape[1]

        frame[row1:row2, col1:col2] = self.avatar

    def extract_tile(self, row, col):
        """
        Extract a tile array from the tiles image.
        """
        row1 = row * TILE_SIZE
        row2 = row1 + TILE_SIZE

        col1 = col * TILE_SIZE
        col2 = col1 + TILE_SIZE

        return self.tiles[row1:row2, col1:col2]

    def move(self):
        """
        Move a customer in the store.
        """
        # Randomly choose from all possible positions for next move
        # choices = self.supermarket.positions[self.section]
        # choice = np.random.choice(len(choices))

        # Set new position
        # if choices[choice][0] < self.supermarket.nrows:
        #    self.row = choices[choice][0]

        # if choices[choice][1] < self.supermarket.ncols:
        #    self.col = choices[choice][1]

        self.row, self.col = self.get_rand_position()

        time.sleep(1)


if __name__ == "__main__":
    background = np.zeros((640, 640, 3), np.uint8)
    tiles = cv2.imread("tiles.png")

    market = SupermarketMap(FLOOR, tiles)
    customer = Customer(market, tiles)  # TODO: Error at 20

    while True:
        frame = background.copy()

        key = cv2.waitKey(1)

        market.draw(frame)
        customer.draw(frame)
        time.sleep(1)
        customer.move()

        if key == 113:  # 'q' key
            break

        cv2.imshow("frame", frame)

    cv2.destroyAllWindows()

    market.write_image("supermarket.png")
