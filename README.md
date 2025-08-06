# Recursive vs Iterative: Finding All Spaces

When you need to **visit every room/space** (not just find one target), there are important differences between recursive and iterative approaches.

## 🔄 Recursive Approach: "Tell Everyone You Meet"

### How it Works
The recursive approach uses function calls to explore spaces. Each function call handles one location and calls itself for deeper exploration.

```python
def visit_all_recursive(room, visited):
    print(f"Visited {room}")
    visited.add(room)
    
    # Visit all connected rooms
    for next_room in connected_rooms[room]:
        if next_room not in visited:
            visit_all_recursive(next_room, visited)  # Go deeper
```

### Exploration Pattern
```
Start → RoomA → RoomB → RoomC (dead end)
                     ← back to RoomB  
                     → RoomD → RoomE (dead end)
                           ← back to RoomD
                     ← back to RoomB
              ← back to RoomA
              → RoomF (and so on...)
```

## 🔁 Iterative Approach: "Keep a Checklist"

### How it Works
The iterative approach uses explicit data structures (stack/queue) to keep track of places to visit.

```python
def visit_all_iterative(start):
    to_visit = [start]
    visited = set()
    
    while to_visit:
        current = to_visit.pop()
        
        if current not in visited:
            print(f"Visited {current}")
            visited.add(current)
            
            # Add all neighbors to checklist
            for next_room in connected_rooms[current]:
                if next_room not in visited:
                    to_visit.append(next_room)
```

### Exploration Pattern
```
Checklist: [Start]
Visit Start → Add [RoomA, RoomF] to checklist
Visit RoomF → Add [RoomG] to checklist  
Visit RoomG → Add [RoomH] to checklist
Visit RoomH → Nothing to add
... continue with remaining items
```

## 🔍 Key Differences for Exploring ALL Spaces

### 1. Exploration Order

**Recursive (DFS-like):**
- Goes **DEEP first**: Start → A → B → C → all the way down
- Then backtracks and tries other branches
- Like exploring one hallway completely before trying another

**Iterative (can be DFS or BFS):**
- **DFS with Stack**: Similar to recursive (deep first)
- **BFS with Queue**: Goes **WIDE first**: Start → all neighbors → their neighbors
- You control the order by choosing stack vs queue

### 2. Memory Usage

**Recursive:**
```
Call Stack grows with depth:
visit(Start)
  visit(RoomA) 
    visit(RoomB)
      visit(RoomC)  ← Stack is 4 levels deep
```

**Iterative:**
```
Your checklist grows with breadth:
to_visit = [RoomF, RoomG, RoomH, RoomI, RoomJ]  ← You control size
```

### 3. Handling Large Spaces

| Scenario | Recursive | Iterative |
|----------|-----------|-----------|
| **Very deep building** (100 floors) | 💥 Stack overflow risk | ✅ Works fine |
| **Very wide building** (1000 rooms per floor) | ✅ Works fine | 🤔 Large queue, but manageable |
| **Huge maze** | 💥 Might crash | ✅ Can handle it |

### 4. Control and Flexibility

**Recursive:**
- Fixed exploration order (DFS)
- Hard to pause/resume exploration
- Difficult to add custom logic between visits

**Iterative:**
- Can easily switch between DFS and BFS
- Easy to pause and resume
- Simple to add custom processing logic
- Can prioritize certain paths (priority queue)

## 🎯 Real-World Examples

### Website Crawler

**Recursive approach:**
```
Visit homepage → Follow first link → Follow its first link → ...
Keep going deeper until you hit a page with no new links
Then backtrack and try other links
```

**Iterative approach:**
```
Add homepage to queue
Visit homepage, add all its links to queue
Visit next page in queue, add its links to queue  
Continue until queue is empty
```

### File System Explorer

**Recursive:**
- Natural for directory traversal
- Simple code structure
- Risk with very deep folder structures

**Iterative:**
- Better for large file systems
- Can implement breadth-first (show all files in current directory first)
- More memory efficient for deep structures

## ⚖️ Pros and Cons

### Recursive Approach

**Pros:**
- ✅ Simple, clean code
- ✅ Natural for tree-like structures
- ✅ Easy to understand the logic
- ✅ Automatic backtracking

**Cons:**
- ❌ Stack overflow risk with deep structures
- ❌ Fixed exploration order (DFS only)
- ❌ Hard to control memory usage
- ❌ Difficult to pause/resume

### Iterative Approach

**Pros:**
- ✅ Can handle very large/deep structures
- ✅ Flexible exploration order (DFS, BFS, custom)
- ✅ Better memory control
- ✅ Easy to pause/resume
- ✅ Production-ready

**Cons:**
- ❌ More complex code
- ❌ Manual state management
- ❌ Need to choose appropriate data structure

## 🛠️ When to Use Which?

### Use Recursive When:
- Space is not too deep (< 1000 levels)
- You want simple, clean code
- Exploring file systems, small networks
- Educational purposes or prototyping
- Tree-like structures with guaranteed depth limits

### Use Iterative When:
- Space might be very deep or very large
- You need to control memory usage
- Building production systems (web crawlers, game AIs)
- You want to switch between DFS and BFS easily
- Need to pause/resume exploration
- Performance is critical

## 🎯 Summary

**Bottom line:** For exploring ALL spaces, iterative gives you more control and safety, while recursive gives you cleaner code but with limitations. Choose based on your specific requirements:

- **Small, controlled spaces** → Recursive for simplicity
- **Large, unknown spaces** → Iterative for safety and control
- **Production systems** → Almost always iterative
- **Learning/prototyping** → Either, but recursive is often easier to understand
