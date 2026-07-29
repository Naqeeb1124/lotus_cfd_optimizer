"""Graph-based CAD compiler for the autonomous CFD design framework.

Classes:
- DesignGraph: NetworkX representation of thermal enclosure geometry.
- GraphCompiler: Compiles a design graph into build123d CAD operations.

Usage:
    graph = DesignGraph()
    graph.add_component("inlet", "Inlet", diameter=150.0)
    graph.add_component("tray1", "Tray", width=400.0, depth=300.0)
    graph.add_component("plenum1", "Plenum", width=400.0, height=100.0)
    graph.add_component("deflector1", "Deflector", angle=45.0)
    graph.add_relationship("inlet", "directs_flow_to", "plenum1")
    graph.add_relationship("plenum1", "supports", "tray1")
    graph.add_relationship("tray1", "adjacent_to", "deflector1")

    compiler = GraphCompiler()
    main_body, boi = compiler.compile(graph)
    export_step(main_body, "main_body.STEP")
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import pathlib
import networkx as nx
from build123d import (
    BuildPart, BuildSketch, Rectangle, Locations, Location, Mode, Align,
    Compound, Cylinder, Box, Plane, extrude, export_step, Axis, Solid, PolarLocations
)
from dataclasses import dataclass, field
from enum import Enum, auto
import math


class ComponentType(Enum):
    """Enumeration of component types in the design graph."""
    INLET = auto()
    OUTLET = auto()
    TRAY = auto()
    PLENUM = auto()
    DEFLECTOR = auto()
    FAN = auto()
    HEATER = auto()


@dataclass
class Component:
    """Represents a CAD component in the design graph."""
    id: str
    type: ComponentType
    params: Dict[str, float] = field(default_factory=dict)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class DesignGraph:
    """NetworkX-based representation of thermal enclosure geometry."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.components: Dict[str, Component] = {}

    def add_component(
        self,
        component_id: str,
        component_type: str,
        **params: float
    ) -> None:
        """Add a component to the design graph.

        Args:
            component_id: Unique identifier for the component.
            component_type: Type of component (e.g., "Inlet", "Tray").
            **params: Component-specific parameters (e.g., diameter=150.0).
        """
        try:
            comp_type = ComponentType[component_type.upper()]
        except KeyError:
            raise ValueError(f"Invalid component type: {component_type}")

        component = Component(
            id=component_id,
            type=comp_type,
            params=params
        )
        self.components[component_id] = component
        self.graph.add_node(component_id, component=component)

    def add_relationship(
        self,
        source_id: str,
        relationship: str,
        target_id: str
    ) -> None:
        """Add a directed relationship between components.

        Args:
            source_id: ID of the source component.
            relationship: Type of relationship (e.g., "directs_flow_to").
            target_id: ID of the target component.
        """
        if source_id not in self.components or target_id not in self.components:
            raise ValueError("Invalid component ID(s)")
        self.graph.add_edge(source_id, target_id, relationship=relationship)

    def get_component(self, component_id: str) -> Optional[Component]:
        """Retrieve a component by ID."""
        return self.components.get(component_id)


class GraphCompiler:
    """Compiles a design graph into build123d CAD operations."""

    def __init__(self):
        self.export_dir = pathlib.Path("E:/Projects/lotus_power/CAD_files/graph_compiler")
        self.export_dir.mkdir(exist_ok=True, parents=True)

    def compile(self, graph: DesignGraph) -> Tuple[Compound, Compound]:
        """Compile a design graph into CAD geometry.

        Args:
            graph: DesignGraph instance.

        Returns:
            Tuple of (main_body, boi) build123d Compounds.
        """
        with BuildPart() as main_body:
            # Traverse components and build CAD
            for component_id, component in graph.components.items():
                self._build_component(component)

            # Apply relationships (e.g., spatial positioning)
            for source_id, target_id, data in graph.graph.edges(data=True):
                self._apply_relationship(
                    graph.get_component(source_id),
                    graph.get_component(target_id),
                    data["relationship"]
                )

        # Generate BOI (Body of Influence)
        boi = self._generate_boi(graph)

        return main_body.part, boi

    def _build_component(self, component: Component) -> None:
        """Build a single component using build123d."""
        if component.type == ComponentType.INLET:
            with Locations(Location((0, 0, component.params.get("z_pos", 0.0)))):
                Cylinder(
                    radius=component.params["diameter"] / 2,
                    height=10.0,
                    align=(Align.CENTER, Align.CENTER, Align.MIN)
                )
        elif component.type == ComponentType.OUTLET:
            with Locations(Location((0, 0, component.params.get("z_pos", 0.0)))):
                Cylinder(
                    radius=component.params["diameter"] / 2,
                    height=10.0,
                    align=(Align.CENTER, Align.CENTER, Align.MIN)
                )
        elif component.type == ComponentType.TRAY:
            with BuildSketch():
                Rectangle(
                    component.params["width"],
                    component.params["depth"]
                )
                Rectangle(
                    component.params["width"] - 2 * component.params["thickness"],
                    component.params["depth"] - 2 * component.params["thickness"],
                    mode=Mode.SUBTRACT
                )
            extrude(amount=-component.params["height"])
        elif component.type == ComponentType.PLENUM:
            with BuildSketch():
                Rectangle(
                    component.params["width"],
                    component.params["depth"]
                )
            extrude(amount=component.params["height"])
        elif component.type == ComponentType.DEFLECTOR:
            angle_rad = math.radians(component.params["angle"])
            with BuildSketch():
                Rectangle(
                    component.params["width"],
                    component.params["thickness"]
                )
            extrude(amount=component.params["height"])
            # Rotate deflector by specified angle
            with Locations(Location((0, 0, component.position[2]))):
                current_part = main_body.part
                current_part = current_part.rotate(Axis.Z, component.params["angle"])
        elif component.type == ComponentType.FAN:
            with Locations(Location((0, 0, component.params.get("z_pos", 0.0)))):
                Box(
                    component.params["width"],
                    component.params["depth"],
                    component.params["height"]
                )
        elif component.type == ComponentType.HEATER:
            with Locations(Location((0, 0, component.params.get("z_pos", 0.0)))):
                Box(
                    component.params["width"],
                    component.params["depth"],
                    component.params["height"]
                )

    def _apply_relationship(
        self,
        source: Component,
        target: Component,
        relationship: str
    ) -> None:
        """Apply spatial relationships between components."""
        if relationship == "directs_flow_to":
            # Position target below source (e.g., inlet → plenum)
            target.position = (
                source.position[0],
                source.position[1],
                source.position[2] - source.params.get("height", 100.0)
            )
        elif relationship == "supports":
            # Position target above source (e.g., plenum → tray)
            target.position = (
                source.position[0],
                source.position[1],
                source.position[2] + source.params.get("height", 100.0)
            )
        elif relationship == "adjacent_to":
            # Position target adjacent to source (e.g., tray → deflector)
            target.position = (
                source.position[0] + source.params.get("width", 200.0) + 10.0,
                source.position[1],
                source.position[2]
            )

    def _generate_boi(self, graph: DesignGraph) -> Compound:
        """Generate Body of Influence (BOI) for meshing."""
        with BuildPart() as boi:
            for component in graph.components.values():
                if component.type == ComponentType.TRAY:
                    Box(
                        component.params["width"] - 4.0,
                        component.params["depth"] - 4.0,
                        50.0,
                        align=(Align.CENTER, Align.CENTER, Align.MIN)
                    ).translate((0, 0, component.position[2]))
                elif component.type == ComponentType.PLENUM:
                    Box(
                        component.params["width"] - 4.0,
                        component.params["depth"] - 4.0,
                        component.params["height"] - 4.0,
                        align=(Align.CENTER, Align.CENTER, Align.MIN)
                    ).translate((0, 0, component.position[2]))
        return boi.part


# ===== Usage Example =====
if __name__ == "__main__":
    # 1. Define a design graph
    graph = DesignGraph()
    graph.add_component("inlet", "Inlet", diameter=150.0, z_pos=200.0)
    graph.add_component("plenum1", "Plenum", width=400.0, depth=300.0, height=100.0)
    graph.add_component("tray1", "Tray", width=400.0, depth=300.0, height=36.5, thickness=1.5)
    graph.add_component("deflector1", "Deflector", width=200.0, height=50.0, thickness=5.0, angle=45.0)
    graph.add_component("outlet", "Outlet", diameter=150.0, z_pos=-200.0)
    
    # Add relationships
    graph.add_relationship("inlet", "directs_flow_to", "plenum1")
    graph.add_relationship("plenum1", "supports", "tray1")
    graph.add_relationship("tray1", "adjacent_to", "deflector1")
    graph.add_relationship("tray1", "directs_flow_to", "outlet")

    # 2. Compile to CAD
    compiler = GraphCompiler()
    main_body, boi = compiler.compile(graph)

    # 3. Export
    export_step(main_body, str(compiler.export_dir / "main_body.STEP"))
    export_step(boi, str(compiler.export_dir / "boi.STEP"))
    print(f"CAD exported to {compiler.export_dir}")