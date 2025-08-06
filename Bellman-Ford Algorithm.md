# Bellman-Ford Algorithm: Complete Guide

The **Bellman-Ford Algorithm** is a **single-source shortest path algorithm** that can handle graphs with **negative edge weights** and **detect negative weight cycles**. Unlike Dijkstra's algorithm, it works correctly even when some edges have negative weights.

## 🎯 Key Features

- ✅ **Handles negative edge weights**
- ✅ **Detects negative weight cycles**
- ✅ **Single-source shortest paths**
- ❌ **Slower than Dijkstra's** (when no negative weights)
- ❌ **Cannot handle negative cycles** (but can detect them)

## 🔄 Core Concept: Edge Relaxation

**Relaxation** is the process of updating the shortest distance to a vertex if a shorter path is found.

```python
def relax(u, v, weight, distances):
    """Relax edge (u, v) with given weight"""
    if distances[u] + weight < distances[v]:
        distances[v] = distances[u] + weight
        return True  # Distance was updated
    return False
```

The key insight: If there are no negative cycles, the shortest path between any two vertices has at most **V-1 edges** (where V is the number of vertices).

## 🧮 Algorithm Steps

### 1. **Initialize Distances**
- Set distance to source = 0
- Set distance to all other vertices = ∞

### 2. **Relax All Edges (V-1 times)**
- For each iteration, relax every edge in the graph
- After V-1 iterations, shortest paths are found (if no negative cycles)

### 3. **Detect Negative Cycles**
- Run one more iteration
- If any distance can still be reduced, there's a negative cycle

## 💻 Implementation

### Basic Bellman-Ford
```python
def bellman_ford(graph, source):
    """
    graph: List of edges [(u, v, weight), ...]
    source: Starting vertex
    Returns: (distances, has_negative_cycle)
    """
    # Get all vertices
    vertices = set()
    for u, v, w in graph:
        vertices.add(u)
        vertices.add(v)
    
    # Step 1: Initialize distances
    distances = {v: float('inf') for v in vertices}
    distances[source] = 0
    
    # Step 2: Relax edges V-1 times
    for i in range(len(vertices) - 1):
        updated = False
        for u, v, weight in graph:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                updated = True
        
        # Early termination optimization
        if not updated:
            break
    
    # Step 3: Check for negative cycles
    has_negative_cycle = False
    for u, v, weight in graph:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            has_negative_cycle = True
            break
    
    return distances, has_negative_cycle

# Example usage
edges = [
    ('A', 'B', -1),
    ('A', 'C', 4),
    ('B', 'C', 3),
    ('B', 'D', 2),
    ('B', 'E', 2),
    ('D', 'B', 1),
    ('D', 'C', 5),
    ('E', 'D', -3)
]

distances, has_cycle = bellman_ford(edges, 'A')
print(f"Distances: {distances}")
print(f"Has negative cycle: {has_cycle}")
```

### Advanced Implementation with Path Tracking
```python
def bellman_ford_with_paths(graph, source):
    """
    Returns distances and actual shortest paths
    """
    vertices = set()
    for u, v, w in graph:
        vertices.add(u)
        vertices.add(v)
    
    distances = {v: float('inf') for v in vertices}
    predecessors = {v: None for v in vertices}
    distances[source] = 0
    
    # Relax edges V-1 times
    for i in range(len(vertices) - 1):
        for u, v, weight in graph:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                predecessors[v] = u
    
    # Check for negative cycles
    negative_cycle_vertices = set()
    for u, v, weight in graph:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            negative_cycle_vertices.add(v)
    
    # Build paths
    def get_path(target):
        if target in negative_cycle_vertices:
            return None  # Path affected by negative cycle
        
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = predecessors[current]
        return path[::-1] if path else None
    
    paths = {v: get_path(v) for v in vertices}
    
    return distances, paths, len(negative_cycle_vertices) > 0

# Example with path tracking
distances, paths, has_cycle = bellman_ford_with_paths(edges, 'A')
for vertex in distances:
    print(f"Path to {vertex}: {paths[vertex]}, Distance: {distances[vertex]}")
```

## 📊 Step-by-Step Example

Let's trace through a simple example:

### Graph:
```
    A --(-1)--> B
    |           |
   (4)         (2)
    |           |
    v           v
    C <--(3)-- D
         ^
        (-3)
         |
         E
```

### Edges: A→B(-1), A→C(4), B→D(2), D→C(3), B→E(2), E→D(-3)

### Execution:

**Initial:** distances = {A: 0, B: ∞, C: ∞, D: ∞, E: ∞}

**Iteration 1:**
- Relax A→B: distances[B] = 0 + (-1) = -1
- Relax A→C: distances[C] = 0 + 4 = 4
- Result: {A: 0, B: -1, C: 4, D: ∞, E: ∞}

**Iteration 2:**
- Relax B→D: distances[D] = -1 + 2 = 1
- Relax B→E: distances[E] = -1 + 2 = 1
- Relax D→C: distances[C] = min(4, 1 + 3) = 4 (no change)
- Result: {A: 0, B: -1, C: 4, D: 1, E: 1}

**Iteration 3:**
- Relax E→D: distances[D] = min(1, 1 + (-3)) = -2
- Result: {A: 0, B: -1, C: 4, D: -2, E: 1}

**Iteration 4:**
- Relax D→C: distances[C] = min(4, -2 + 3) = 1
- Final: {A: 0, B: -1, C: 1, D: -2, E: 1}

**Negative Cycle Check:** No further improvements possible ✅

## 🔄 Comparison with Other Algorithms

| Algorithm | Time Complexity | Space | Negative Weights | Negative Cycles | Best Use Case |
|-----------|----------------|-------|------------------|----------------|---------------|
| **Bellman-Ford** | O(VE) | O(V) | ✅ Yes | ✅ Detects | Negative weights, cycle detection |
| **Dijkstra** | O((V+E)logV) | O(V) | ❌ No | ❌ No | Positive weights only |
| **Floyd-Warshall** | O(V³) | O(V²) | ✅ Yes | ✅ Detects | All-pairs shortest paths |
| **Johnson's** | O(V²logV + VE) | O(V²) | ✅ Yes | ✅ Detects | Sparse graphs, all-pairs |

## 🎮 Real-World Applications

### 1. Currency Exchange Arbitrage
```python
def detect_arbitrage(exchange_rates):
    """
    Detect arbitrage opportunities in currency exchange
    exchange_rates: dict of {(from_currency, to_currency): rate}
    """
    # Convert to negative logarithms (multiplication becomes addition)
    import math
    
    edges = []
    currencies = set()
    
    for (from_curr, to_curr), rate in exchange_rates.items():
        # Use negative log to find arbitrage (positive cycles become negative)
        weight = -math.log(rate)
        edges.append((from_curr, to_curr, weight))
        currencies.add(from_curr)
        currencies.add(to_curr)
    
    # Run Bellman-Ford from any currency
    start_currency = next(iter(currencies))
    distances, has_negative_cycle = bellman_ford(edges, start_currency)
    
    return has_negative_cycle  # Negative cycle = arbitrage opportunity

# Example: USD → EUR → GBP → USD
exchange_rates = {
    ('USD', 'EUR'): 0.85,
    ('EUR', 'GBP'): 0.90,
    ('GBP', 'USD'): 1.35,
    ('USD', 'GBP'): 0.75,
}

arbitrage_exists = detect_arbitrage(exchange_rates)
print(f"Arbitrage opportunity: {arbitrage_exists}")
```

### 2. Network Routing with QoS
```python
def network_routing_with_penalties(network, source, penalties):
    """
    Find shortest paths considering network penalties
    penalties: negative values for premium links, positive for congested links
    """
    edges = []
    for (u, v), base_cost in network.items():
        penalty = penalties.get((u, v), 0)
        total_cost = base_cost + penalty
        edges.append((u, v, total_cost))
    
    distances, has_cycle = bellman_ford(edges, source)
    
    if has_cycle:
        print("Warning: Network has inconsistent routing policies!")
    
    return distances

# Network with premium/penalty links
network = {
    ('Router1', 'Router2'): 10,
    ('Router1', 'Router3'): 15,
    ('Router2', 'Router4'): 12,
    ('Router3', 'Router4'): 8,
}

penalties = {
    ('Router1', 'Router3'): -5,  # Premium link (negative penalty)
    ('Router2', 'Router4'): 3,   # Congested link (positive penalty)
}

routing_costs = network_routing_with_penalties(network, 'Router1', penalties)
```

### 3. Transit Route Planning with Constraints
```python
def transit_planning_with_subsidies(routes, subsidies, start_station):
    """
    Plan transit routes considering subsidies (negative costs) and penalties
    """
    edges = []
    
    for (from_station, to_station, mode), base_cost in routes.items():
        # Apply subsidies for certain routes (e.g., government incentives)
        subsidy = subsidies.get((from_station, to_station, mode), 0)
        net_cost = base_cost - subsidy  # Subsidy reduces cost
        edges.append((from_station, to_station, net_cost))
    
    distances, has_negative_cycle = bellman_ford(edges, start_station)
    
    if has_negative_cycle:
        print("Infinite subsidy loop detected! Free travel possible.")
        return None
    
    return distances

# Transit network with subsidies
routes = {
    ('StationA', 'StationB', 'bus'): 5,
    ('StationB', 'StationC', 'train'): 8,
    ('StationC', 'StationA', 'metro'): 6,
    ('StationA', 'StationC', 'bus'): 12,
}

subsidies = {
    ('StationA', 'StationB', 'bus'): 2,    # $2 subsidy
    ('StationB', 'StationC', 'train'): 3,  # $3 subsidy
    ('StationC', 'StationA', 'metro'): 10, # Large subsidy (potential issue)
}

costs = transit_planning_with_subsidies(routes, subsidies, 'StationA')
```

## ⚡ Optimizations

### 1. Early Termination
```python
def bellman_ford_optimized(graph, source):
    vertices = set()
    for u, v, w in graph:
        vertices.add(u)
        vertices.add(v)
    
    distances = {v: float('inf') for v in vertices}
    distances[source] = 0
    
    # Early termination if no updates in iteration
    for i in range(len(vertices) - 1):
        updated = False
        for u, v, weight in graph:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                updated = True
        
        if not updated:  # No changes, can terminate early
            print(f"Converged early at iteration {i + 1}")
            break
    
    # Check for negative cycles
    has_negative_cycle = False
    for u, v, weight in graph:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            has_negative_cycle = True
            break
    
    return distances, has_negative_cycle
```

### 2. Queue-Based (SPFA - Shortest Path Faster Algorithm)
```python
from collections import deque

def spfa(graph, source):
    """
    SPFA: Queue-based optimization of Bellman-Ford
    Average case: O(E), Worst case: still O(VE)
    """
    # Build adjacency list
    adj = {}
    vertices = set()
    for u, v, w in graph:
        vertices.add(u)
        vertices.add(v)
        if u not in adj:
            adj[u] = []
        adj[u].append((v, w))
    
    distances = {v: float('inf') for v in vertices}
    in_queue = {v: False for v in vertices}
    count = {v: 0 for v in vertices}  # Count times vertex is relaxed
    
    distances[source] = 0
    queue = deque([source])
    in_queue[source] = True
    
    while queue:
        u = queue.popleft()
        in_queue[u] = False
        
        if u in adj:
            for v, weight in adj[u]:
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    count[v] += 1
                    
                    # Negative cycle detection
                    if count[v] >= len(vertices):
                        return None, True  # Negative cycle found
                    
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True
    
    return distances, False
```

## 🚨 Important Considerations

### Negative Cycle Detection
```python
def find_negative_cycle(graph):
    """
    Find and return vertices affected by negative cycles
    """
    # Run Bellman-Ford from all vertices to ensure we catch all cycles
    all_vertices = set()
    for u, v, w in graph:
        all_vertices.add(u)
        all_vertices.add(v)
    
    affected_vertices = set()
    
    for source in all_vertices:
        distances, has_cycle = bellman_ford(graph, source)
        
        if has_cycle:
            # Find which vertices are affected
            # Run additional iterations to propagate negative infinity
            for _ in range(len(all_vertices)):
                for u, v, weight in graph:
                    if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                        distances[v] = distances[u] + weight
                        affected_vertices.add(v)
    
    return list(affected_vertices)

def mark_negative_infinity(graph, source):
    """
    Mark vertices reachable from negative cycles as -∞
    """
    distances, has_cycle = bellman_ford(graph, source)
    
    if not has_cycle:
        return distances
    
    # Run V more iterations to propagate negative infinity
    vertices = set()
    for u, v, w in graph:
        vertices.add(u)
        vertices.add(v)
    
    for _ in range(len(vertices)):
        for u, v, weight in graph:
            if distances[u] == float('-inf'):
                distances[v] = float('-inf')
            elif distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = float('-inf')
    
    return distances
```

### Handling Unreachable Vertices
```python
def bellman_ford_complete(graph, source):
    """
    Complete implementation handling all edge cases
    """
    vertices = set()
    for u, v, w in graph:
        vertices.add(u)
        vertices.add(v)
    
    # Initialize
    distances = {v: float('inf') for v in vertices}
    predecessors = {v: None for v in vertices}
    distances[source] = 0
    
    # Standard Bellman-Ford
    for i in range(len(vertices) - 1):
        for u, v, weight in graph:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                predecessors[v] = u
    
    # Detect and handle negative cycles
    negative_cycle_vertices = set()
    for u, v, weight in graph:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            negative_cycle_vertices.add(v)
    
    # Propagate negative infinity
    if negative_cycle_vertices:
        for _ in range(len(vertices)):
            for u, v, weight in graph:
                if u in negative_cycle_vertices or distances[u] == float('-inf'):
                    distances[v] = float('-inf')
                    negative_cycle_vertices.add(v)
    
    # Classify vertices
    result = {}
    for v in vertices:
        if distances[v] == float('inf'):
            result[v] = ('unreachable', None, None)
        elif distances[v] == float('-inf'):
            result[v] = ('negative_cycle', None, None)
        else:
            # Build path
            path = []
            current = v
            while current is not None:
                path.append(current)
                current = predecessors[current]
            result[v] = ('reachable', distances[v], path[::-1])
    
    return result
```

## 🔍 Debugging and Testing

### Test Cases
```python
def test_bellman_ford():
    # Test 1: Simple positive weights (should match Dijkstra)
    test1 = [('A', 'B', 1), ('B', 'C', 2), ('A', 'C', 4)]
    dist1, cycle1 = bellman_ford(test1, 'A')
    assert dist1['C'] == 3 and not cycle1
    
    # Test 2: Negative weights but no cycle
    test2 = [('A', 'B', -1), ('B', 'C', 2), ('A', 'C', 4)]
    dist2, cycle2 = bellman_ford(test2, 'A')
    assert dist2['C'] == 1 and not cycle2
    
    # Test 3: Negative cycle
    test3 = [('A', 'B', 1), ('B', 'C', -3), ('C', 'A', 1)]
    dist3, cycle3 = bellman_ford(test3, 'A')
    assert cycle3
    
    # Test 4: Disconnected graph
    test4 = [('A', 'B', 1), ('C', 'D', 1)]
    dist4, cycle4 = bellman_ford(test4, 'A')
    assert dist4['D'] == float('inf')
    
    print("All tests passed!")

test_bellman_ford()
```

## 🎯 When to Use Bellman-Ford

### ✅ Use Bellman-Ford When:
- **Negative edge weights** are present
- **Negative cycle detection** is needed
- **Robust solution** is required (handles all cases)
- **Simple implementation** is preferred over performance
- **All edge cases** need to be handled correctly

### ❌ Consider Alternatives When:
- **All weights are non-negative** → Use Dijkstra's algorithm
- **Performance is critical** and no negative weights → Use Dijkstra's
- **All-pairs shortest paths** needed → Consider Floyd-Warshall
- **Very dense graphs** → Johnson's algorithm might be better

## 🏆 Advanced Topics

### Bellman-Ford in Distributed Systems
```python
def distributed_bellman_ford(node_id, neighbors, initial_distance):
    """
    Simulate distributed Bellman-Ford (used in routing protocols)
    Each node maintains its own distance vector
    """
    # Distance vector: {destination: (cost, next_hop)}
    distance_vector = {node_id: (0, node_id)}  # Distance to self is 0
    
    def update_from_neighbor(neighbor_id, neighbor_vector):
        updated = False
        for destination, (neighbor_cost, _) in neighbor_vector.items():
            if destination == node_id:
                continue
                
            # Cost through this neighbor
            cost_via_neighbor = neighbors[neighbor_id] + neighbor_cost
            
            if (destination not in distance_vector or 
                cost_via_neighbor < distance_vector[destination][0]):
                distance_vector[destination] = (cost_via_neighbor, neighbor_id)
                updated = True
        
        return updated
    
    return update_from_neighbor

# Example: Node A with neighbors B(cost 1) and C(cost 2)
node_a = distributed_bellman_ford('A', {'B': 1, 'C': 2}, 0)

# Receive update from neighbor B
neighbor_b_vector = {'B': (0, 'B'), 'D': (3, 'D')}
updated = node_a('B', neighbor_b_vector)
```

## 🎯 Summary

The **Bellman-Ford Algorithm** is an essential tool for shortest path problems when:

- **Handling negative weights**: Unlike Dijkstra's, it correctly processes negative edge weights
- **Detecting negative cycles**: Crucial for many real-world applications like arbitrage detection
- **Ensuring correctness**: Provides robust solutions even with complex edge weight scenarios
- **Implementing distributed systems**: Forms the basis of distance-vector routing protocols

**Key Trade-off**: Sacrifices the speed of Dijkstra's O((V+E)logV) for the flexibility to handle negative weights and detect negative cycles with O(VE) complexity.

**Best Practice**: Use Dijkstra's for non-negative weights (faster), use Bellman-Ford when you need to handle negative weights or detect negative cycles.
