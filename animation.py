import time

import cv2
import numpy as np
import pandas as pd
from faker import Faker

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


class Supermarket:
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
        self.customers = []
        self.tprobs = self.load_tprobs()
        self.entry_times = self.load_entry_times()
        self.last_id = 0
        self.current_time = self.entry_times.loc[0, "timestamp"]
        self.register = pd.DataFrame(columns=["timestamp", "customer_no", "location"])

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

        mapping = {
            "E": "entrance",
            "F": "fruit",
            "D": "dairy",
            "S": "spices",
            "B": "drinks",
            "C": "checkout",
            "#": "wall",
        }
        # Create empty dict to store possible positions
        self.positions = {mapping[key]: [] for key in chars}

        for row, line in enumerate(self.contents):
            for col, char in enumerate(line):
                # Save all possible positions of letters in a dict
                self.positions[mapping[char]].append((row, col))

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

    def load_tprobs(self):
        """
        Load transition probabilities from a CSV file.
        """
        return pd.read_csv("data/transition_probabilities.csv", index_col=[0])

    def load_entry_times(self):
        """
        Load entry times from a CSV file.
        """
        return pd.read_csv("data/entry_times.csv", index_col=[0])

    def add_new_customer(self, new_customer):
        """
        Add one customer.
        """
        assert isinstance(new_customer, Customer)
        self.customers.append(new_customer)

        # Register customer action
        self.register_action(self.current_time, new_customer.id, new_customer.section)

        print(f"{new_customer.name} (ID: {new_customer.id}) entered the supermarket.")

    def add_new_customers(self, num_customers):
        """
        Add multiple new customers.
        """
        for _ in range(num_customers):
            f = Faker()
            new_customer = Customer(self.last_id + 1, f.name(), market, tiles)
            self.add_new_customer(new_customer)
            self.last_id += 1

    def remove_customer(self, customer):
        """
        Remove customer from store.
        """
        assert isinstance(customer, Customer)
        self.customers.remove(customer)
        print(f"{customer.name} (ID: {customer.id}) has left the store.")

    def register_action(self, timestamp, id, section):
        """
        Register an action for the CSV output at the end.
        """
        self.register.loc[len(self.register)] = [timestamp, id, section]

    def write_image(self, filename):
        """
        Writes the image into a file.
        """
        cv2.imwrite(filename, self.image)


class Customer:
    """
    A single customer that moves through the supermarket
    in a MCMC simulation.
    """

    def __init__(self, id, name, supermarket, tiles, section="entrance"):
        """
        supermarket: A SuperMarketMap object
        avatar : a numpy array containing a 32x32 tile image
        """
        self.id = id
        self.name = name
        self.supermarket = supermarket
        self.row, self.col = self.get_rand_position("entrance")
        self.tiles = tiles
        self.avatar = self.extract_tile(7, 0)
        self.section = section

        time.sleep(1)

    def __repr__(self) -> str:
        return f"<Customer {self.name}, currently in section '{self.section}'>"

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

    def next_section(self, tprobs=None):
        """
        Propagates the customer to the next state.
        Returns nothing.
        """
        if tprobs == None:
            tprobs = self.supermarket.tprobs

        current = self.section

        self.section = np.random.choice(tprobs.columns, p=tprobs.loc[self.section])

        self.row, self.col = self.get_rand_position()

        if current == self.section:
            print(f"{self.name} (ID: {self.id}) stayed in {current}.")
        else:
            print(
                f"{self.name} (ID: {self.id}) moved from {current} to {self.section}."
            )

    def is_active(self):
        """
        Returns True if the customer has not reached the checkout yet.
        """

        return self.section != "checkout"


if __name__ == "__main__":
    tiles = cv2.imread("tiles.png")

    # Instantiate Supermarket object
    market = Supermarket(FLOOR, tiles)

    # Create background of the same size as the supermarket
    background = np.zeros(market.image.shape, np.uint8)

    i = 0

    while True:
        # Create a new frame
        frame = background.copy()

        # Look for key actions
        key = cv2.waitKey(1)

        # Draw the Supermarket
        market.draw(frame)

        market.current_time = market.entry_times.loc[i, "timestamp"]

        for c in market.customers:
            # Move customers currently in the store to next section
            c.next_section()
            market.register_action(market.current_time, c.id, c.section)

            # Draw the customer
            c.draw(frame)

            # Remove customers that reached checkout
            if not c.is_active():
                market.remove_customer(c)

        # Add new customers entering the store
        market.add_new_customers(market.entry_times.loc[i, "new_customers"])

        print(f"Currently there are {len(market.customers)} customers in the store.\n")

        i += 1

        if i == len(market.entry_times):
            break

        if key == 113:  # 'q' key
            break

        cv2.imshow("frame", frame)

    # Save register to CSV file
    market.register.to_csv("data/simulation_results_anim.csv")

    cv2.destroyAllWindows()

    market.write_image("supermarket.png")
