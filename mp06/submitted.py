# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)

from collections import deque
from collections import defaultdict
from queue import PriorityQueue
import math

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function
    rows = maze.size.y
    cols = maze.size.x
    q = deque([maze.start])
    visited = set()
    visited.add(maze.start)
    BFSPath = {}
    BFSPath[maze.start] = (-1, -1)
    while q:
        cur = q.popleft()
        if cur == maze.waypoints[0]:
            break
        for nei in maze.neighbors_all(cur[0], cur[1]):
            if nei not in visited:
                visited.add(nei)
                q.append(nei)
                BFSPath[nei] = cur
    path = []
    while BFSPath[cur] != (-1, -1):
        path.insert(0, cur)
        cur = BFSPath[cur]
    path.insert(0, cur)
    return path

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single
    def chebyshev(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    
    q = PriorityQueue()
    h = chebyshev(maze.start, maze.waypoints[0])
    q.put((h, (0, maze.start)))
    visited = set([maze.start])
    astarPath = {maze.start: (-1, -1)}
    while q:
        f, (g, cur) = q.get()
        if cur == maze.waypoints[0]:
            break
        for nei in maze.neighbors_all(cur[0], cur[1]):
            if nei not in visited:
                visited.add(nei)
                h_cur = chebyshev(nei, maze.waypoints[0])
                q.put((g + 1 + h_cur, (g + 1, nei)))
                astarPath[nei] = cur
    path = []
    while astarPath[cur] != (-1, -1):
        path.insert(0, cur)
        cur = astarPath[cur]
    path.insert(0, cur)
    return path

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    def chebyshev(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    
    def build_MST(waypoints):
        total = 0
        if not waypoints:
            return total
        edges = PriorityQueue()
        node = waypoints[0]
        visited_node = set()
        visited_node.add(node)
        for nei in waypoints:
            if nei not in visited_node:
                dis = chebyshev(node, nei)
                edges.put((dis, (node, nei)))
        while edges and len(visited_node) < len(waypoints):
            dis, (a, b) = edges.get()
            if b not in visited_node:
                visited_node.add(b)
                total += dis
                for nei in waypoints:
                    if nei not in visited_node:
                        dis = chebyshev(b, nei)
                        edges.put((dis, (b, nei)))
        return total

    def nearest_wp(cur, waypoints):
        if not waypoints:
            return 0
        return min([chebyshev(cur, wp) for wp in waypoints])
    
    not_visited_wp = set(maze.waypoints)
    visited_wp = set()
    q = PriorityQueue()
    mst = build_MST(list(not_visited_wp))
    h = nearest_wp(maze.start, maze.waypoints) + mst
    q.put((h, (0, (mst, maze.start))))
    visited_node = set([maze.start])
    astarPath = {}
    astarPath[maze.start] = (-1, -1)
    
    final_path = [maze.start]
    while q:
        f, (g, (tmp_mst, cur)) = q.get()
        print(f, (g, (tmp_mst, cur)))
        if cur in maze.waypoints and cur in not_visited_wp:
            # print(cur)
            path = []
            tmp = cur
            while astarPath[tmp] != (-1, -1):
                path.insert(0, tmp)
                tmp = astarPath[tmp]
            final_path += path
            astarPath = {}
            astarPath[cur] = (-1, -1)
            visited_node.clear()
            visited_node.add(cur)
            visited_wp.add(cur)
            if len(visited_wp) == len(maze.waypoints):
                break
            not_visited_wp.remove(cur)
            mst = build_MST(list(not_visited_wp))
            q = PriorityQueue()
        for nei in maze.neighbors_all(cur[0], cur[1]):
            if nei not in visited_node:
                visited_node.add(nei)
                if nei not in not_visited_wp:
                    h_cur = nearest_wp(nei, not_visited_wp) + mst
                    new_mst = mst
                else:
                    tmp_set = not_visited_wp - set([nei])
                    new_mst = build_MST(list(tmp_set))
                    h_cur = nearest_wp(nei, tmp_set) + new_mst
                astarPath[nei] = cur
                # print(cur)
                # print((g+ 1 + h_cur, (g + 1, nei)))
                q.put((g+ 1 + h_cur, (g + 1, (new_mst, nei))))
    return final_path