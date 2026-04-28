from __future__ import annotations

from collections import deque

from .search_world import Node, SearchWorld


def bfs(world: SearchWorld) -> list[Node]:
    q = deque([world.start])
    parent: dict[Node, Node | None] = {world.start: None}
    while q:
        cur = q.popleft()
        if cur == world.goal:
            break
        for nxt in world.neighbors(cur):
            if nxt not in parent:
                parent[nxt] = cur
                q.append(nxt)
    if world.goal not in parent:
        return []
    path = []
    n = world.goal
    while n is not None:
        path.append(n)
        n = parent[n]
    return list(reversed(path))
