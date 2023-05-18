import numpy as np
import pandas as pd
from faker import Faker


class Customer:
    """
    A single customer that moves through the supermarket
    in a MCMC simulation.
    """

    def __init__(self, id, name, section="entrance"):
        self.id = id
        self.name = name
        self.section = section

    def __repr__(self):
        return f"<Customer {self.name}, currently in section '{self.section}'>"

    def next_section(self, tprobs):
        """
        Propagates the customer to the next state.
        Returns nothing.
        """
        current = self.section

        self.section = np.random.choice(tprobs.columns, p=tprobs.loc[self.section])

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


class Supermarket:
    """
    Class that manages multiple Customer instances that
    are currently in the market.
    """

    def __init__(self, name):
        self.name = name
        self.customers = []
        self.tprobs = self.load_tprobs()
        self.entry_times = self.load_entry_times()
        self.last_id = 0
        self.current_time = 0
        self.register = pd.DataFrame(columns=["timestamp", "customer_no", "location"])

    def __repr__(self):
        return f"Supermarket class. The name of the supermarket is {self.name}"

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
            new_customer = Customer(self.last_id + 1, f.name())
            self.add_new_customer(new_customer)
            self.last_id += 1

    def remove_customer(self, customer):
        """
        Remove customer from store.
        """
        assert isinstance(customer, Customer)
        self.customers.remove(customer)
        print(f"{customer.name} (ID: {customer.id}) has left the store.")

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

    def register_action(self, timestamp, id, section):
        """
        Register an action for the CSV output at the end.
        """
        self.register.loc[len(self.register)] = [timestamp, id, section]

    def run(self):
        """
        Run the script.
        """
        # Loop through all the lines of entry times
        for i in range(len(self.entry_times)):
            print(self.entry_times.loc[i, "timestamp"])
            self.current_time = self.entry_times.loc[i, "timestamp"]

            for c in self.customers:
                # Move customers currently in the store to next section
                c.next_section(self.tprobs)
                self.register_action(self.current_time, c.id, c.section)

                # Remove customers that reached checkout
                if not c.is_active():
                    self.remove_customer(c)

            # Add new customers entering the store
            self.add_new_customers(self.entry_times.loc[i, "new_customers"])

            print(
                f"Currently there are {len(self.customers)} customers in the store.\n"
            )

        # Save register to CSV file
        self.register.to_csv("data/simulation_results.csv")


def main():
    # Instantiate a supermarket object
    netto = Supermarket("Netto")

    # Run the simulation
    netto.run()


if __name__ == "__main__":
    main()
