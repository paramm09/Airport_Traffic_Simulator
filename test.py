from backend.data.airport_graph import build_airport_graph
from backend.algorithms.dijkstra import dijkstra

graph = build_airport_graph()

print(dijkstra(graph, "G1", "R3"))
print(dijkstra(graph, "G2", "R2"))
print(dijkstra(graph, "G5", "R1"))
print(dijkstra(graph, "G10", "R2"))
print(dijkstra(graph, "G6", "R4"))