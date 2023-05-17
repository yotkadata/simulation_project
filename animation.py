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
        # Split the floor string into a two dimensional matrix
        self.contents = [list(row) for row in floor.split("\n")]
        self.ncols = len(self.contents[0])
        self.nrows = len(self.contents)
        self.image = np.zeros(
            (self.nrows * TILE_SIZE, self.ncols * TILE_SIZE, 3), dtype=np.uint8
        )
        self.prepare_map()

    def extract_tile(self, row, col):
        """
        Extract a tile array from the tiles image.
        """
        row1 = row * TILE_SIZE
        row2 = row1 + TILE_SIZE

        col1 = col * TILE_SIZE
        col2 = col1 + TILE_SIZE

        return self.tiles[row1:row2, col1:col2]

    def get_tile(self, char):
        """
        Return the array for a given tile character.
        """
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
        """
        Prepare the entire image as a big numpy array.
        """
        # Get all unique characters in floor plan
        chars = "".join(set(FLOOR)).replace("\n", "")

        # Create empty dict to store possible positions
        self.positions = {key: [] for key in chars}

        for row, line in enumerate(self.contents):
            for col, char in enumerate(line):
                # Save all possible positions of letters in a dict
                self.positions[char].append((row, col))

                # Get tile for current position
                bm = self.get_tile(char)

                # Calculate window size and position to insert tile
                row1 = row * TILE_SIZE
                row2 = row1 + TILE_SIZE

                col1 = col * TILE_SIZE
                col2 = col1 + TILE_SIZE

                # Add tile to image
                self.image[row1:row2, col1:col2] = bm

    def draw(self, frame):
        """
        Draw the image into a frame.
        """
        frame[0 : self.image.shape[0], 0 : self.image.shape[1]] = self.image

    def write_image(self, filename):
        """
        Writes the image into a file.
        """
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

        time.sleep(1)

    def __repr__(self) -> str:
        return f"Customer object"

    def get_rand_position(self, section=None):
        """
        Randomly select a position in a given section.
        """
        if section == None:
            section = self.section

        # Get all possible positions in the section
        choices = self.supermarket.positions[section]

        # Randomly choose one and return it
        i = np.random.choice(len(choices))

        return choices[i]

    def draw(self, frame):
        """
        Add the customer image to the frame.
        """
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
        self.row, self.col = self.get_rand_position()


if __name__ == "__main__":
    tiles = cv2.imread("tiles.png")

    # Instantiate Supermarket object
    market = SupermarketMap(FLOOR, tiles)

    # Instantiate customer object
    customer = Customer(market, tiles)

    # Create background of the same size as the supermarket
    background = np.zeros(market.image.shape, np.uint8)

    while True:
        # Create a new frame
        frame = background.copy()

        # Look for key actions
        key = cv2.waitKey(1)

        # Draw the Supermarket
        market.draw(frame)

        # Draw the customer
        customer.draw(frame)
        time.sleep(1)

        # Move the customer
        customer.move()

        if key == 113:  # 'q' key
            break

        cv2.imshow("frame", frame)

    cv2.destroyAllWindows()

    market.write_image("supermarket.png")
