from .base import LocalSearch


class NoLocalSearch(LocalSearch):
    def optimize(self, route):
        return