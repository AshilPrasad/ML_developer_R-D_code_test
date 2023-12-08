import random

class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.adjacency_list = {vertex: [] for vertex in range(vertices)}
        self.generate_random_connections()

    def add_edge(self, source, destination):
        self.adjacency_list[source].append(destination)
        self.adjacency_list[destination].append(source)

    def generate_random_connections(self):
        # Generating random connections between nodes
        for vertex in range(self.vertices):
            num_connections = random.randint(1, min(3, self.vertices - 1))  # Limiting the number of connections
            possible_destinations = [v for v in range(self.vertices) if v != vertex]
            connections = random.sample(possible_destinations, num_connections)
            for destination in connections:
                self.add_edge(vertex, destination)

    def __str__(self):
        graph_str = ""
        for vertex, connections in self.adjacency_list.items():
            graph_str += f"{vertex} -> {connections}\n"
        return graph_str

# Example usage
num_vertices = 6
random_graph = Graph(num_vertices)

# Display the random graph
print("Random Graph:")
print(random_graph)
