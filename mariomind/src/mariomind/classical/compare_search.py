from __future__ import annotations

from .astar import astar
from .bfs import bfs
from .search_world import SearchWorld


def compare() -> dict:
    world = SearchWorld()
    bfs_path = bfs(world)
    astar_path = astar(world)
    return {
        "bfs_path_len": len(bfs_path),
        "astar_path_len": len(astar_path),
        "goal_reached_bfs": bool(bfs_path),
        "goal_reached_astar": bool(astar_path),
    }
