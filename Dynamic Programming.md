# Dynamic Programming: Complete Guide

**Dynamic Programming (DP)** is an **algorithmic technique for solving complex problems by breaking them down into smaller subproblems**. It dramatically improves efficiency by storing and reusing previously calculated results.

## 🧩 Core Concepts

### "Same problem, don't solve twice!"

**Basic Idea:**
- Large problem = combination of smaller problems
- Store solutions to small problems (memoization)
- Reuse stored solutions when needed

## 🔑 DP Requirements

### 1. **Optimal Substructure**
- Optimal solution of the larger problem = combination of optimal solutions of smaller problems

### 2. **Overlapping Subproblems**
- Same smaller problems appear multiple times

## 📊 Simple Example: Fibonacci Sequence

### Naive Recursion (Inefficient)
```python
def fibonacci_naive(n):
    if n <= 1:
        return n
    return fibonacci_naive(n-1) + fibonacci_naive(n-2)

# When calling fibonacci(5):
#     fib(5)
#    /      \
#  fib(4)  fib(3)
#  /   \    /   \
# fib(3) fib(2) fib(2) fib(1)
# ...         ...
# fib(2) is calculated multiple times!
```

**Problem:** fib(2), fib(3), etc. are calculated redundantly

### DP Solution 1: Memoization (Top-Down)
```python
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
        
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]
```

### DP Solution 2: Tabulation (Bottom-Up)
```python
def fibonacci_dp(n):
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]
```

**Performance Difference:**
- Naive recursion: O(2^n) - very slow
- DP: O(n) - very fast

## 🎯 DP in Pathfinding

### Minimum Cost Path in Grid
```python
def min_path_cost(grid):
    rows, cols = len(grid), len(grid[0])
    
    # dp[i][j] = minimum cost from (0,0) to (i,j)
    dp = [[0] * cols for _ in range(rows)]
    
    # Initialize
    dp[0][0] = grid[0][0]
    
    # First row
    for j in range(1, cols):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    
    # First column
    for i in range(1, rows):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    
    # Fill remaining cells
    for i in range(1, rows):
        for j in range(1, cols):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
    
    return dp[rows-1][cols-1]

# Example:
grid = [
    [1, 3, 1],
    [1, 5, 1], 
    [4, 2, 1]
]
# Minimum cost path: 1→1→1→1→1 = 5
```

## 💰 Classic DP Problems

### 1. Coin Change Problem
```python
def coin_change(coins, amount):
    # dp[i] = minimum number of coins to make amount i
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Example: coins = [1, 3, 4], amount = 6
# Answer: 2 (3 + 3)
```

### 2. Knapsack Problem
```python
def knapsack(weights, values, capacity):
    n = len(weights)
    # dp[i][w] = maximum value considering first i items with capacity w
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't take the i-th item
            dp[i][w] = dp[i-1][w]
            
            # Take the i-th item (if possible)
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], 
                    dp[i-1][w - weights[i-1]] + values[i-1])
    
    return dp[n][capacity]
```

### 3. Longest Common Subsequence (LCS)
```python
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    # dp[i][j] = length of LCS for text1[:i] and text2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# Example: "abcde", "ace" → 3 ("ace")
```

### 4. Edit Distance (Levenshtein Distance)
```python
def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    # dp[i][j] = minimum operations to convert word1[:i] to word2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters
    for j in range(n + 1):
        dp[0][j] = j  # Insert all characters
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # No operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Delete
                    dp[i][j-1],    # Insert
                    dp[i-1][j-1]   # Replace
                )
    
    return dp[m][n]
```

## 🚌 DP in Transit Route Planning

### Optimal Transit Route with Time Windows
```python
def optimal_transit_route(stations, timetable, start_time, start_station, target_station):
    # dp[station][time] = minimum cost to reach station at time
    dp = {}
    
    # Initial state
    dp[(start_station, start_time)] = 0
    
    for current_time in range(start_time, end_time):
        for station in stations:
            if (station, current_time) not in dp:
                continue
                
            current_cost = dp[(station, current_time)]
            
            # Check all possible next moves
            for route in timetable:
                if (route.from_station == station and 
                    route.departure_time >= current_time):
                    
                    arrival_key = (route.to_station, route.arrival_time)
                    new_cost = current_cost + route.cost
                    
                    if arrival_key not in dp or dp[arrival_key] > new_cost:
                        dp[arrival_key] = new_cost
    
    # Find minimum cost to target station
    min_cost = float('inf')
    for (station, time), cost in dp.items():
        if station == target_station:
            min_cost = min(min_cost, cost)
    
    return min_cost if min_cost != float('inf') else -1
```

### Multi-Modal Transportation
```python
def multi_modal_transit(graph, start, end, max_transfers):
    # dp[station][transfers_used] = minimum time to reach station with transfers_used
    dp = {}
    dp[(start, 0)] = 0
    
    for transfers in range(max_transfers + 1):
        for station in graph.stations:
            if (station, transfers) not in dp:
                continue
                
            current_time = dp[(station, transfers)]
            
            # Try each transportation mode
            for mode in ['bus', 'subway', 'walk']:
                for neighbor, travel_time in graph.get_connections(station, mode):
                    new_transfers = transfers + (1 if mode != 'walk' else 0)
                    
                    if new_transfers <= max_transfers:
                        new_time = current_time + travel_time
                        key = (neighbor, new_transfers)
                        
                        if key not in dp or dp[key] > new_time:
                            dp[key] = new_time
    
    # Find minimum time to reach end station
    result = float('inf')
    for transfers in range(max_transfers + 1):
        if (end, transfers) in dp:
            result = min(result, dp[(end, transfers)])
    
    return result if result != float('inf') else -1
```

## 🔄 DP Approaches Comparison

### Memoization vs Tabulation

| | **Memoization** | **Tabulation** |
|---|---|---|
| **Approach** | Top-Down (Recursion) | Bottom-Up (Iteration) |
| **Memory** | Call stack + Cache | Table only |
| **Computation** | Only what's needed | All subproblems |
| **Implementation** | More intuitive | More systematic |
| **Stack Overflow** | Possible with deep recursion | No risk |

### Example: Both Approaches for LCS
```python
# Memoization (Top-Down)
def lcs_memo(text1, text2, i=0, j=0, memo={}):
    if (i, j) in memo:
        return memo[(i, j)]
    
    if i == len(text1) or j == len(text2):
        return 0
    
    if text1[i] == text2[j]:
        result = 1 + lcs_memo(text1, text2, i+1, j+1, memo)
    else:
        result = max(lcs_memo(text1, text2, i+1, j, memo),
                    lcs_memo(text1, text2, i, j+1, memo))
    
    memo[(i, j)] = result
    return result

# Tabulation (Bottom-Up)
def lcs_tab(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

## ⚡ DP Optimization Techniques

### 1. Space Optimization
```python
# Before: O(n) space
def fibonacci_dp(n):
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# After: O(1) space
def fibonacci_optimized(n):
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1
```

### 2. State Compression with Bitmasks
```python
def traveling_salesman(dist, n):
    # mask: bitmask representing visited cities
    # dp[mask][i] = minimum cost to visit cities in mask and end at city i
    dp = {}
    
    def solve(mask, pos):
        if mask == (1 << n) - 1:  # All cities visited
            return dist[pos][0]    # Return to start
        
        if (mask, pos) in dp:
            return dp[(mask, pos)]
        
        result = float('inf')
        for next_city in range(n):
            if mask & (1 << next_city) == 0:  # Unvisited city
                new_mask = mask | (1 << next_city)
                cost = dist[pos][next_city] + solve(new_mask, next_city)
                result = min(result, cost)
        
        dp[(mask, pos)] = result
        return result
    
    return solve(1, 0)  # Start from city 0
```

### 3. Rolling Array Technique
```python
def unique_paths(m, n):
    # Only need previous row to compute current row
    prev = [1] * n
    
    for i in range(1, m):
        curr = [1] * n
        for j in range(1, n):
            curr[j] = curr[j-1] + prev[j]
        prev = curr
    
    return prev[n-1]
```

## 🎮 Real-World Applications

### Gaming
- **Experience Optimization**: Minimum time/cost to level up
- **Resource Management**: Maximum effect with limited resources
- **Skill Trees**: Optimal path to target skills
- **Game Economy**: Dynamic pricing strategies

### Economics/Finance
- **Investment Portfolios**: Maximum return for given risk
- **Dynamic Pricing**: Optimal pricing strategies over time
- **Inventory Management**: Cost minimization
- **Option Pricing**: Black-Scholes models

### AI/Machine Learning
- **Sequence Alignment**: DNA, protein, text comparison
- **Optimal Policies**: Bellman equations in reinforcement learning
- **Neural Networks**: Backpropagation algorithms
- **Natural Language Processing**: Viterbi algorithm for HMMs

### Operations Research
- **Supply Chain**: Optimal distribution strategies
- **Production Planning**: Resource allocation over time
- **Network Flow**: Maximum flow problems
- **Scheduling**: Optimal task assignments

## ⚖️ Advantages and Disadvantages

### Advantages ✅
- **Dramatic Time Improvement**: Exponential → Polynomial complexity
- **Guaranteed Optimal Solution**: With correct recurrence relation
- **Intuitive Approach**: Combines solutions of smaller problems
- **Wide Applicability**: Many optimization problems

### Disadvantages ❌
- **Memory Usage**: Stores all subproblem solutions
- **Design Difficulty**: Proper state definition is crucial
- **Overhead**: May be excessive for simple problems
- **Space Complexity**: Can be prohibitive for large state spaces

## 🔍 DP Problem-Solving Steps

### 1. **Problem Analysis**
- Does it have optimal substructure?
- Are there overlapping subproblems?
- Can we define states clearly?

### 2. **State Definition**
- What information do we need to track?
- dp[i], dp[i][j], dp[mask], etc.

### 3. **Recurrence Relation**
- How does current state relate to previous states?
- Base cases and transitions

### 4. **Base Cases**
- What are the simplest subproblems?
- Initialize correctly

### 5. **Computation Order**
- Bottom-up: ensure dependencies are computed first
- Top-down: handle with memoization

### 6. **Space Optimization** (if needed)
- Can we reduce space complexity?
- Rolling arrays, state compression

## 🎯 When to Use DP

### ✅ DP is Suitable When:
- **Optimization problems** (finding maximum/minimum)
- **Counting problems** (number of ways to do something)
- **Decision problems** with optimal choice sequences
- **Problems with recursive structure** and overlapping subproblems
- **Multi-stage decision processes**

### ❌ DP is Not Suitable When:
- **Greedy algorithms** work (no overlapping subproblems)
- **Memory is extremely limited**
- **Real-time constraints** with no preprocessing time
- **Problems without optimal substructure**
- **Simple problems** where overhead exceeds benefits

## 🏆 Advanced DP Patterns

### 1. Interval DP
```python
def matrix_chain_multiplication(dimensions):
    n = len(dimensions) - 1
    # dp[i][j] = minimum scalar multiplications for matrices i to j
    dp = [[0] * n for _ in range(n)]
    
    for length in range(2, n + 1):  # length of chain
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            
            for k in range(i, j):
                cost = (dp[i][k] + dp[k+1][j] + 
                       dimensions[i] * dimensions[k+1] * dimensions[j+1])
                dp[i][j] = min(dp[i][j], cost)
    
    return dp[0][n-1]
```

### 2. Tree DP
```python
def max_path_sum_tree(root):
    def dfs(node):
        if not node:
            return 0
        
        left_gain = max(dfs(node.left), 0)
        right_gain = max(dfs(node.right), 0)
        
        # Maximum path through current node
        current_max = node.val + left_gain + right_gain
        
        # Update global maximum
        nonlocal max_sum
        max_sum = max(max_sum, current_max)
        
        # Return maximum gain from this subtree
        return node.val + max(left_gain, right_gain)
    
    max_sum = float('-inf')
    dfs(root)
    return max_sum
```

### 3. Digit DP
```python
def count_numbers_with_property(n):
    s = str(n)
    
    def dp(pos, tight, started, state):
        if pos == len(s):
            return 1 if started and satisfies_property(state) else 0
        
        if (pos, tight, started, state) in memo:
            return memo[(pos, tight, started, state)]
        
        limit = int(s[pos]) if tight else 9
        result = 0
        
        for digit in range(0, limit + 1):
            new_tight = tight and (digit == limit)
            new_started = started or (digit > 0)
            new_state = update_state(state, digit) if new_started else state
            
            result += dp(pos + 1, new_tight, new_started, new_state)
        
        memo[(pos, tight, started, state)] = result
        return result
    
    memo = {}
    return dp(0, True, False, initial_state)
```

## 🎯 Summary

Dynamic Programming is a **"smart memory"** technique that efficiently solves complex problems by:

- **Breaking down problems**: Divide complex problems into simpler subproblems
- **Storing solutions**: Remember previously computed results (avoid redundant work)
- **Combining optimally**: Build optimal solutions from optimal substructure
