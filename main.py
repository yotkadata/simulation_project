"""
Simulation of clients moving in a supermarket.
Based on Markov Chains
"""

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

TILES = cv2.imread("img/tiles.png")


class Supermarket:
    """
    Simulates and visualizes the movement of clients in a supermarket.
    """

    def __init__(self, name, floor=FLOOR, tiles=TILES):
        """
        name: str containing the name of the supermarket
        floor: str with each character representing a tile
        tiles: numpy array containing all the tile images
        """
        self.name = name
        self.tiles = tiles
        self.floor_matrix = self.split_floor(floor)
        self.image = self.prepare_image()
        self.customers = []
        self.last_id = 0
        self.current_time = self.entry_times.loc[0, "timestamp"]
        self.register = None
        self.positions_taken = []

    @property
    def tprobs(self):
        """
        Load transition probabilities from a CSV file.
        """
        return pd.read_csv("data/transition_probabilities.csv", index_col=[0])

    @property
    def entry_times(self):
        """
        Load entry times from a CSV file.
        """
        return pd.read_csv("data/entry_times.csv", index_col=[0])

    @property
    def positions(self):
        """
        Calculate all possible positions by section.
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
        positions = {mapping[key]: [] for key in chars}

        for row, line in enumerate(self.floor_matrix):
            for col, char in enumerate(line):
                # Save all possible positions of letters in a dict
                positions[mapping[char]].append((row, col))

        return positions

    def prepare_image(self):
        """
        Prepare the entire image as a big numpy array.
        """

        # Create an empty array of the image size to be filled later
        image = np.zeros(
            (
                len(self.floor_matrix) * TILE_SIZE,
                len(self.floor_matrix[0]) * TILE_SIZE,
                4,
            ),
            dtype=np.uint8,
        )

        for row, line in enumerate(self.floor_matrix):
            for col, char in enumerate(line):
                # Get tile for current position
                current_tile = self.get_tile(char)

                # Add alpha channel
                rgba = cv2.cvtColor(current_tile, cv2.COLOR_RGB2RGBA)

                # Calculate window size and position to insert tile
                row1 = row * TILE_SIZE
                row2 = row1 + TILE_SIZE

                col1 = col * TILE_SIZE
                col2 = col1 + TILE_SIZE

                # Add tile to image
                image[row1:row2, col1:col2] = rgba

        return image

    def split_floor(self, floor):
        """
        Split the floor string into a two dimensional matrix.
        """
        return [list(row) for row in floor.split("\n")]

    def extract_tile(self, position):
        """
        Extract a tile array from the tiles image.
        """
        row1 = position[0] * TILE_SIZE
        row2 = row1 + TILE_SIZE

        col1 = position[1] * TILE_SIZE
        col2 = col1 + TILE_SIZE

        return self.tiles[row1:row2, col1:col2]

    def get_tile(self, character):
        """
        Return the array for a given tile character.
        """

        tile_position = {
            "#": (0, 0),
            "E": (0, 1),
            "F": (0, 2),
            "S": (0, 3),
            "D": (0, 4),
            "B": (1, 0),
            "C": (1, 1),
            "others": (1, 2),
        }

        if character not in tile_position:
            character = "others"

        return self.extract_tile(tile_position[character])

    def draw(self, frame):
        """
        Draw the image into a frame.
        """
        frame[0 : self.image.shape[0], 0 : self.image.shape[1]] = self.image

    def add_new_customer(self, new_customer):
        """
        Add one customer to the list of customers currently in the store.
        """
        assert isinstance(new_customer, Customer)
        self.customers.append(new_customer)

        # Register customer action
        self.register_action(
            self.current_time, new_customer.cid, new_customer.name, new_customer.section
        )

        print(f"{new_customer.name} (ID: {new_customer.cid}) entered the supermarket.")

    def add_new_customers(self, num_customers):
        """
        Create multiple new customer objects and add them to list of customers in the market.
        """
        for _ in range(num_customers):
            faker = Faker()
            self.last_id += 1
            new_customer = Customer(self.last_id, faker.name(), self)
            self.add_new_customer(new_customer)

    def remove_customer(self, customer):
        """
        Remove customer from store.
        """
        assert isinstance(customer, Customer)
        self.customers.remove(customer)
        print(f"{customer.name} (ID: {customer.cid}) has left the store.")

    def register_action(self, timestamp, customer_id, name, section):
        """
        Register an action for the CSV output at the end.
        """
        if not isinstance(self.register, pd.DataFrame):
            self.register = pd.DataFrame(
                columns=["timestamp", "customer_no", "name", "location"]
            )

        self.register.loc[len(self.register)] = [timestamp, customer_id, name, section]

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

            for customer in self.customers:
                # Move customers currently in the store to next section
                customer.next_section(self.tprobs)
                self.register_action(
                    self.current_time, customer.cid, customer.name, customer.section
                )

                # Remove customers that reached checkout
                if not customer.is_active():
                    self.remove_customer(customer)

            # Add new customers entering the store
            self.add_new_customers(self.entry_times.loc[i, "new_customers"])

            print(
                f"Currently there are {len(self.customers)} customers in the store.\n"
            )

        # Save register to CSV file
        csv_path = "data/simulation_results.csv"
        self.register.to_csv(csv_path)
        print(f"Results saved to {csv_path}.")

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

            for customer in self.customers:
                # Move customers currently in the store to next section
                customer.next_section()

            # Add new customers entering the store
            self.add_new_customers(self.entry_times.loc[i, "new_customers"])

            for customer in self.customers:
                # Draw the customer
                customer.draw(frame)

                # Remove customers that reached checkout
                if not customer.is_active():
                    self.remove_customer(customer)

            # Reset taken positions
            self.positions_taken = []

            print(
                f"Currently there are {len(self.customers)} customers in the store.\n"
            )

            i += 1

            if i == steps:
                break

            if key == 113:  # 'q' key
                break

            time.sleep(0.5)

            cv2.imshow("frame", frame)

        cv2.destroyAllWindows()

        cv2.imwrite("supermarket.png", self.image)


class Customer:
    """
    A single customer that moves through the supermarket
    in a MCMC simulation.
    """

    def __init__(self, cid, name, supermarket, section="entrance"):
        """
        supermarket: A SuperMarketMap object
        avatar : a numpy array containing a 32x32 tile image
        """
        self.cid = cid
        self.name = name
        self.supermarket = supermarket
        self.position = self.get_rand_position("entrance")
        self.section = section

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

        while True:
            # Randomly choose one position
            i = np.random.choice(len(choices))

            # Make sure the position is not taken by another customer
            if choices[i] not in self.supermarket.positions_taken:
                self.supermarket.positions_taken.append(choices[i])
                return choices[i]

    def draw(self, frame):
        """
        Add the customer image to the frame.
        """
        row1 = self.position[0] * TILE_SIZE
        row2 = row1 + self.avatar.shape[0]

        col1 = self.position[1] * TILE_SIZE
        col2 = col1 + self.avatar.shape[1]

        frame[row1:row2, col1:col2] = self.avatar

    @property
    def avatar(self):
        """
        Generate a random avatar.
        """
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

        self.position = self.get_rand_position()

        if current == self.section:
            print(f"{self.name} (ID: {self.cid}) stayed in {current}.")
        else:
            print(
                f"{self.name} (ID: {self.cid}) moved from {current} to {self.section}."
            )

    def is_active(self):
        """
        Returns True if the customer has not reached the checkout yet.
        """

        return self.section != "checkout"


def main() -> None:
    """
    Main function.
    """
    # Instantiate a supermarket object
    netto = Supermarket("Netto")

    # Run the simulation
    # netto.simulate()

    # Run the animation
    netto.animate()


if __name__ == "__main__":
    main()
