from .base import BaseLocalSearch


class NoLocalSearch(BaseLocalSearch):
    def optimize(self, route):
        route.update_state()
        return