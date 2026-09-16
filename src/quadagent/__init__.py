"""QuadAgent - a minimal deep agent for 2D quad meshing with Gmsh.

Lecture baseline: a single-agent ReAct loop (no subagents) with the three
deep-agent ingredients - a planning tool (`write_todos`), a virtual
filesystem scoped to the session workspace, and domain tools that execute
arbitrary Gmsh code in a sandbox and extract mesh quality metrics.
"""

__version__ = "0.1.0"
