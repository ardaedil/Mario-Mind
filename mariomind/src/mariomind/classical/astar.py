from __future__ import annotations

import heapq

from .search_world import Node, SearchWorld


def astar(world: SearchWorld) -> list[Node]:
    frontier = [(world.heuristic(world.start), 0, world.start)]
    parent: dict[Node, Node | None] = {world.start: None}
    g = {world.start: 0}
    while frontier:
        _, cost, cur = heapq.heappop(frontier)
        if cur == world.goal:
            break
        for nxt in world.neighbors(cur):
            new_cost = cost + 1
            if nxt not in g or new_cost < g[nxt]:
                g[nxt] = new_cost
                parent[nxt] = cur
                f = new_cost + world.heuristic(nxt)
                heapq.heappush(frontier, (f, new_cost, nxt))
    if world.goal not in parent:
        return []
    path = []
    n = world.goal
    while n is not None:
        path.append(n)
        n = parent[n]
    return list(reversed(path))
