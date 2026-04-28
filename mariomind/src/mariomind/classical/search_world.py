from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    x: int


class SearchWorld:
    def __init__(self, length: int = 20, gaps: set[int] | None = None, blocked: set[int] | None = None):
        self.length = length
        self.start = Node(0)
        self.goal = Node(length - 1)
        self.gaps = {5, 12} if gaps is None else gaps
        self.blocked = {9} if blocked is None else blocked

    def neighbors(self, node: Node) -> list[Node]:
        moves = [node.x + 1, node.x + 2]
        out = []
        for x in moves:
            if 0 <= x < self.length and x not in self.blocked and x not in self.gaps:
                out.append(Node(x))
        return out

    def heuristic(self, node: Node) -> float:
        return self.goal.x - node.x
