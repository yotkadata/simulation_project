import time

import cv2
import numpy as np
import pandas as pd
import py_avataaars as pa
from faker import Faker

TILE_SIZE = 32

FLOOR = """
########################################
########################################
##BBBBBBBBDDDDDDDDDDSSSSSSSSSSFFFFFFFF##
##BBBBBBBBDDDDDDDDDDSSSSSSSSSSFFFFFFFF##
##BBBBBBBBDDDDDDDDDDSSSSSSSSSSFFFFFFFF##
##BBBBBBBBDDDDDDDDDDSSSSSSSSSSFFFFFFFF##
##BBBBBB####DDDDDD####SSSSSS####FFFFFF##
##BBBBBB####DDDDDD####SSSSSS####FFFFFF##
##BBBBBB####DDDDDD####SSSSSS####FFFFFF##
##BBBBBB####DDDDDD####SSSSSS####FFFFFF##
##BBBBBB####DDDDDD####SSSSSS####FFFFFF##
##BBBBBB####DDDDDD####SSSSSS####FFFFFF##
##BBBBBB####DDDDDD####SSSSSS####FFFFFF##
##BBBBBB####DDDDDD####SSSSSS####FFFFFF##
##BBBBBB####DDDDDD####SSSSSS####FFFFFF##
##BBBBBB####DDDDDD####SSSSSS####FFFFFF##
##BBBBBB####DDDDDD####SSSSSS####FFFFFF##
##BBBBBB####DDDDDD####SSSSSS####FFFFFF##
##BBBBBB####DDDDDD####SSSSSS####FFFFFF##
##BBBBBB####DDDDDD####SSSSSS####FFFFFF##
BBBBBBBBBBDDDDDDDDDDSSSSSSSSEEEEEEEEEEEE
BBBBBBBBBBDDDDDDDDDDSSSSSSSSEEEEEEEEEEEE
####CCCC####CCCC####CCCC####EEEEEEEEEEEE
####CCCC####CCCC####CCCC####EEEEEEEEEEEE
####CCCC####CCCC####CCCC####EEEEEEEEEEEE
####CCCC####CCCC####CCCC####EEEEEEEEEEEE
####CCCC####CCCC####CCCC####EEEEEEEEEEEE
####CCCC####CCCC####CCCC####EEEEEEEEEEEE
CCCCCCCCCCCCCCCCCCCCCCCCCCCCEEEEEEEEEEEE
CCCCCCCCCCCCCCCCCCCCCCCCCCCCEEEEEEEEEEEE
""".strip()

TILES = cv2.imread("tiles.png")


class Supermarket:
    """
    Visualizes the supermarket background.
    """

    def __init__(self, name, floor=FLOOR, tiles=TILES):
        """
        floor: a string with each character representing a tile
        tiles: a numpy array containing all the tile images
        """
        self.name = name
        self.tiles = tiles
        # Split the floor string into a two dimensional matrix
        self.contents = [list(row) for row in floor.split("\n")]
        self.ncols = len(self.contents[0])
        self.nrows = len(self.contents)
        self.image = np.zeros(
            (self.nrows * TILE_SIZE, self.ncols * TILE_SIZE, 4), dtype=np.uint8
        )
        self.prepare_map()
        self.customers = []
        self.tprobs = self.load_tprobs()
        self.entry_times = self.load_entry_times()
        self.last_id = 0
        self.current_time = self.entry_times.loc[0, "timestamp"]
        self.register = pd.DataFrame(
            columns=["timestamp", "customer_no", "name", "location"]
        )

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

                # Add alpha channel
                rgba = cv2.cvtColor(bm, cv2.COLOR_RGB2RGBA)

                # Calculate window size and position to insert tile
                row1 = row * TILE_SIZE
                row2 = row1 + TILE_SIZE

                col1 = col * TILE_SIZE
                col2 = col1 + TILE_SIZE

                # Add tile to image
                self.image[row1:row2, col1:col2] = rgba

    def draw(self, frame):
        """
        Draw the image into a frame.
        """
        frame[0 : self.image.shape[0], 0 : self.image.shape[1]] = self.image

    def draw_customers(self, customers, frame):
        """
        Draw all customers in their current position.
        """
        for c in customers:
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
        self.register_action(
            self.current_time, new_customer.id, new_customer.name, new_customer.section
        )

        print(f"{new_customer.name} (ID: {new_customer.id}) entered the supermarket.")

    def add_new_customers(self, num_customers):
        """
        Add multiple new customers.
        """
        for _ in range(num_customers):
            f = Faker()
            self.last_id += 1
            new_customer = Customer(self.last_id, f.name(), self, self.tiles)
            self.add_new_customer(new_customer)

    def remove_customer(self, customer):
        """
        Remove customer from store.
        """
        assert isinstance(customer, Customer)
        self.customers.remove(customer)
        print(f"{customer.name} (ID: {customer.id}) has left the store.")

    def register_action(self, timestamp, id, name, section):
        """
        Register an action for the CSV output at the end.
        """
        self.register.loc[len(self.register)] = [timestamp, id, name, section]

    def simulate(self, steps=None):
        """
        Run the script.
        """
        if steps is None:
            steps = len(self.entry_times)

        # Loop through all the lines of entry times
        for i in range(steps):
            print(self.entry_times.loc[i, "timestamp"])
            self.current_time = self.entry_times.loc[i, "timestamp"]

            for c in self.customers:
                # Move customers currently in the store to next section
                c.next_section(self.tprobs)
                self.register_action(self.current_time, c.id, c.name, c.section)

                # Remove customers that reached checkout
                if not c.is_active():
                    self.remove_customer(c)

            # Add new customers entering the store
            self.add_new_customers(self.entry_times.loc[i, "new_customers"])

            print(
                f"Currently there are {len(self.customers)} customers in the store.\n"
            )

        # Save register to CSV file
        csv_path = "data/simulation_results.csv"
        self.register.to_csv(csv_path)
        print(f"Results saved to {csv_path}.")

    def write_image(self, filename):
        """
        Writes the image into a file.
        """
        cv2.imwrite(filename, self.image)

    def animate(self, steps=None):
        """
        Animate customer movements in the store.
        """
        if steps is None:
            steps = len(self.entry_times)

        # Create background of the same size as the supermarket
        background = np.zeros(self.image.shape, np.uint8)

        i = 0

        while True:
            # Create a new frame
            frame = background.copy()

            # Look for key actions
            key = cv2.waitKey(1)

            # Draw the Supermarket
            self.draw(frame)

            self.current_time = self.entry_times.loc[i, "timestamp"]

            for c in self.customers:
                # Move customers currently in the store to next section
                c.next_section()

            # Add new customers entering the store
            self.add_new_customers(self.entry_times.loc[i, "new_customers"])

            for c in self.customers:
                # Draw the customer
                c.draw(frame)

                # Remove customers that reached checkout
                if not c.is_active():
                    self.remove_customer(c)

            print(
                f"Currently there are {len(self.customers)} customers in the store.\n"
            )

            i += 1

            if i == len(self.entry_times):
                break

            if key == 113:  # 'q' key
                break

            time.sleep(1)

            cv2.imshow("frame", frame)

        cv2.destroyAllWindows()

        self.write_image("supermarket.png")


class Customer:
    """
    A single customer that moves through the supermarket
    in a MCMC simulation.
    """

    def __init__(self, id, name, supermarket, tiles=None, section="entrance"):
        """
        supermarket: A SuperMarketMap object
        avatar : a numpy array containing a 32x32 tile image
        """
        if not isinstance(tiles, np.ndarray):
            tiles = supermarket.tiles

        self.id = id
        self.name = name
        self.supermarket = supermarket
        self.row, self.col = self.get_rand_position("entrance")
        self.tiles = tiles
        self.section = section
        self.avatar = self.generate_avatar()

    def __repr__(self) -> str:
        return f"<Customer {self.name}, currently in section '{self.section}'>"

    def get_rand_position(self, section=None):
        """
        Randomly select a position in a given section.
        """
        if section is None:
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

    def generate_avatar(self):
        avatar = pa.PyAvataaar(
            skin_color=np.random.choice(list(pa.SkinColor)),
            hair_color=np.random.choice(list(pa.HairColor)),
            facial_hair_type=np.random.choice(list(pa.FacialHairType)),
            facial_hair_color=np.random.choice(list(pa.HairColor)),
            top_type=np.random.choice(list(pa.TopType)),
            hat_color=np.random.choice(list(pa.Color)),
            mouth_type=np.random.choice(list(pa.MouthType)),
            eye_type=np.random.choice(list(pa.EyesType)),
            eyebrow_type=np.random.choice(list(pa.EyebrowType)),
            nose_type=np.random.choice(list(pa.NoseType)),
            clothe_type=np.random.choice(list(pa.ClotheType)),
            clothe_color=np.random.choice(list(pa.Color)),
            clothe_graphic_type=np.random.choice(list(pa.ClotheGraphicType)),
            background_color=np.random.choice(list(pa.Color)),
        )

        image_bytes = avatar.render_png()
        image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        image_array = cv2.resize(image_array, (TILE_SIZE, TILE_SIZE))

        return image_array

    def next_section(self, tprobs=None):
        """
        Propagates the customer to the next state.
        Returns nothing.
        """
        if not isinstance(tprobs, pd.DataFrame):
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


def main():
    # Instantiate a supermarket object
    netto = Supermarket("Netto")

    # Run the simulation
    # netto.simulate()

    # Run the animation
    netto.animate()


if __name__ == "__main__":
    main()
