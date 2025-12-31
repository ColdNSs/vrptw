import os


class SolomonInstance:
    """
    A simple data class to hold the VRPTW instance data.
    """

    def __init__(self, name, num_vehicles, capacity, nodes):
        self.name = name
        self.num_vehicles = num_vehicles
        self.capacity = capacity
        # nodes is a list of dictionaries, where index 0 is the depot
        self.nodes = nodes

    def __repr__(self):
        return (f"SolomonInstance(name='{self.name}', "
                f"vehicles={self.num_vehicles}, capacity={self.capacity}, "
                f"num_nodes={len(self.nodes)})")


class Node:
    """
    Wrapper for node data.
    """
    def __init__(self, node_id, x, y, demand, ready_time, due_date, service_time):
        self.id = node_id
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time

    def __index__(self):
        """
        Node can be used as numpy index.
        """
        return self.id

    def __repr__(self):
        return f"Node {self.id}(coord=({self.x}, {self.y}), window=[{self.ready_time}, {self.due_date}], demand={self.demand}, service={self.service_time})"


def read_solomon_instance(filepath):
    """
    Parses a standard Solomon VRPTW text file.

    Args:
        filepath (str): Path to the instance file.

    Returns:
        SolomonInstance: Object containing instance data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Instance file not found: {filepath}")

    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Typically, the first line is the instance name
    instance_name = lines[0]

    # Variables to hold parsed data
    num_vehicles = 0
    capacity = 0
    nodes = []

    # Simple state machine or section tracking
    section = None

    for line in lines[1:]:
        # Detect sections
        if line.startswith("VEHICLE"):
            section = "VEHICLE"
            continue
        elif line.startswith("CUSTOMER"):
            section = "CUSTOMER"
            continue

        # Parse data based on section
        if section == "VEHICLE":
            # Expecting line: NUMBER CAPACITY
            # There is a header line "NUMBER CAPACITY", we skip headers by checking if it's digit
            parts = line.split()
            if not parts[0].isdigit():
                continue  # Skip header line inside VEHICLE section

            num_vehicles = int(parts[0])
            capacity = float(parts[1])

        elif section == "CUSTOMER":
            # Expecting: CUST NO. XCOORD. YCOORD. DEMAND READY TIME DUE DATE SERVICE TIME
            parts = line.split()
            if not parts[0].isdigit():
                continue  # Skip header line inside CUSTOMER section

            # Parse node attributes
            node = Node(
                node_id=int(parts[0]),
                x=float(parts[1]),
                y=float(parts[2]),
                demand=float(parts[3]),
                ready_time=float(parts[4]),
                due_date=float(parts[5]),
                service_time=float(parts[6])
            )
            nodes.append(node)

    return SolomonInstance(instance_name, num_vehicles, capacity, nodes)


# Quick test block
if __name__ == "__main__":
    test_path = "../data/benchmarks/solomon-100/c101.txt"

    try:
        instance = read_solomon_instance(test_path)
        print("Successfully parsed:")
        print(instance)
        print(f"Depot: {instance.nodes[0]}")
        print(f"First Customer: {instance.nodes[1]}")
    except Exception as e:
        print(f"Error: {e}")