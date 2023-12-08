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

class InferenceRules:
    @staticmethod
    def inference(output):
        threshold = 0.5  # Example threshold value
        if output > threshold:
            return "Greater Than Threshold"
        elif output < threshold:
            return "Less Than Threshold"
        else:
            return "Equal to Threshold"

# Example usage
num_vertices = 6
random_graph = Graph(num_vertices)

# Display the random graph
print("Random Graph:")
print(random_graph)

# Perform inference on node 0
node_0_output = random.uniform(0, 1)  # Example: random value between 0 and 1

# Perform inference and decide the next node based on rules
decision = InferenceRules.inference(node_0_output)

# Display the result
print(f"\nInference Result for Node 0: {node_0_output}")
print(f"Decision based on Rules: {decision}")

# Decide the next node based on the decision
next_node = None
if decision == "Greater Than Threshold":
    next_node = random.choice(random_graph.adjacency_list[0])
elif decision == "Less Than Threshold":
    # Choose a random node, excluding those connected to node 0
    possible_next_nodes = set(range(num_vertices)) - set(random_graph.adjacency_list[0])
    next_node = random.choice(list(possible_next_nodes))
else:
    # Choose a random node from those connected to node 0
    next_node = random.choice(random_graph.adjacency_list[0])

print(f"Next Node based on Decision: {next_node}")
