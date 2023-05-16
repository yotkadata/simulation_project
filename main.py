import numpy as np
import pandas as pd


class Customer:
    """
    A single customer that moves through the supermarket
    in a MCMC simulation.
    """

    def __init__(self, id, name, t_probs, section="entrance"):
        self.id = id
        self.name = name
        self.t_probs = t_probs
        self.section = section

    def __repr__(self):
        return f"<Customer {self.name} (ID: {self.id}), currently in section '{self.section}'>"

    def next_section(self):
        """
        Propagates the customer to the next state.
        Returns nothing.
        """
        current = self.section
        self.section = np.random.choice(
            self.t_probs.columns, p=self.t_probs.loc[self.section]
        )

        if self.is_active():
            print(f"{self.name} moved from {current} to {self.section}.")
        else:
            print(f"{self.name} left the store.")  # TODO: Remove customer

    def is_active(self):
        """
        Returns True if the customer has not reached the checkout yet.
        """

        return self.section != "checkout"
