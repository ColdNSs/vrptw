class Route:
    def __init__(self, instance, dist_matrix):
        self.capacity = instance.capacity
        self.depot = instance.nodes[0]
        self.dist_matrix = dist_matrix

        self.sequence = [self.depot]
        self.load = 0.0
        self.time = 0.0
        self.cost = 0.0
        self.finish_time = 0.0
        self.tw_penalties = 0.0
        self.is_closed = False

    @property
    def last_node(self):
        return self.sequence[-1]

    def is_load_feasible(self, node):
        if self.load + node.demand > self.capacity:
            return False
        return True

    def is_time_feasible(self, node):
        prev_node = self.last_node
        dist = self.dist_matrix[prev_node][node]
        departure_time = max(self.time, prev_node.ready_time) + prev_node.service_time
        arrival_time = departure_time + dist

        if arrival_time > node.due_date:
            return False
        return True

    def is_feasible(self, node):
        if not self.is_load_feasible(node):
            return False
        if not self.is_time_feasible(node):
            return False
        return True

    def add_node(self, node):
        prev_node = self.last_node
        dist = self.dist_matrix[prev_node][node]
        self.load += node.demand

        departure_from_prev = max(self.time, prev_node.ready_time) + prev_node.service_time
        arrival_at_new = departure_from_prev + dist

        self.time = max(arrival_at_new, node.ready_time)

        self.sequence.append(node)

    def close_route(self):
        if self.is_closed:
            return
        if self.is_feasible(self.depot):
            self.add_node(self.depot)
        else:
            self.add_node(self.depot)
        self.is_closed = True

    def update_state(self):
        seq = self.sequence

        total_dist = 0.0
        current_time = seq[0].ready_time
        tw_penalties = 0.0

        for k in range(len(seq) - 1):
            prev_node = seq[k]
            next_node = seq[k + 1]

            # 1. Distance
            dist = self.dist_matrix[prev_node][next_node]
            total_dist += dist

            # 2. Time simulation
            departure_time = current_time + prev_node.service_time
            arrival_time = departure_time + dist

            # 3. Time Window Penalty (Late arrival)
            if arrival_time > next_node.due_date:
                # Penalty is proportional to how late we are
                tw_penalties += (arrival_time - next_node.due_date)

            # Update current time (we must wait if we arrive before ready_time)
            current_time = max(arrival_time, next_node.ready_time)

        # Update cost, finish time and time window penalties
        self.cost = total_dist
        self.finish_time = current_time
        self.tw_penalties = tw_penalties

    def __repr__(self):
        path_ids = [n.id for n in self.sequence]
        return f"Route(Load={self.load}, Cost={self.cost:.2f}, TW_Penalties={self.tw_penalties:.2f}, Path={path_ids})"
