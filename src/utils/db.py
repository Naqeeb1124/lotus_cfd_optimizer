"""SQLite database utilities for the autonomous CFD design framework.

Tables:
- designs: Stores parametric design configurations.
- simulations: Records CFD simulation results.
- critiques: Human or rule-based feedback on designs.
- archive: MAP-Elites behavioral archive.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

# Database path
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "lotus.db"
DB_PATH.parent.mkdir(exist_ok=True, parents=True)

# Schema SQL
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS designs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    parameters    TEXT    NOT NULL,  -- JSON string of design variables
    graph_blob    BLOB,              -- Optional pickled NetworkX graph
    created_at    TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS simulations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    design_id     INTEGER NOT NULL,
    mesh_path     TEXT    NOT NULL,  -- Path to .h5 mesh file
    solver_path   TEXT,              -- Optional path to solver log
    mean_cov      REAL,              -- Velocity CoV (average of all shelves)
    pressure_drop REAL,              -- ΔP (Pa) between inlet & outlet
    objective     REAL,              -- Scalar objective for optimization
    status        TEXT    NOT NULL,  -- SUCCESS / FAILED / ABORTED
    created_at    TEXT    DEFAULT (datetime('now')),
    FOREIGN KEY (design_id) REFERENCES designs(id)
);

CREATE TABLE IF NOT EXISTS critiques (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    design_id     INTEGER NOT NULL,
    simulation_id INTEGER,           -- May be NULL (pre-simulation critique)
    critique      TEXT    NOT NULL,  -- Free-form text or JSON payload
    created_at    TEXT    DEFAULT (datetime('now')),
    FOREIGN KEY (design_id)     REFERENCES designs(id),
    FOREIGN KEY (simulation_id) REFERENCES simulations(id)
);

CREATE TABLE IF NOT EXISTS archive (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    cell_id       TEXT    NOT NULL,  -- Discretized behavioral key
    design_id     INTEGER NOT NULL,
    behavior      TEXT    NOT NULL,  -- JSON representing behavioral descriptor
    fitness       REAL    NOT NULL,  -- Objective value (lower = better)
    created_at    TEXT    DEFAULT (datetime('now')),
    FOREIGN KEY (design_id) REFERENCES designs(id)
);
"""

def get_connection() -> sqlite3.Connection:
    """Initialize DB with schema and return a connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_SQL)
    return conn

def insert_design(name: str, params: Dict[str, Any], graph_blob: Optional[bytes] = None) -> int:
    """Insert a design and return its ID."""
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO designs (name, parameters, graph_blob) VALUES (?, ?, ?)",
            (name, json.dumps(params), graph_blob)
        )
        return cursor.lastrowid

def insert_simulation(
    design_id: int,
    mesh_path: str,
    solver_path: Optional[str],
    mean_cov: float,
    pressure_drop: float,
    objective: float,
    status: str
) -> int:
    """Insert a simulation and return its ID."""
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO simulations
               (design_id, mesh_path, solver_path, mean_cov, pressure_drop, objective, status)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (design_id, mesh_path, solver_path, mean_cov, pressure_drop, objective, status)
        )
        return cursor.lastrowid

def insert_critique(design_id: int, simulation_id: Optional[int], critique: str) -> int:
    """Insert a critique and return its ID."""
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO critiques (design_id, simulation_id, critique) VALUES (?, ?, ?)",
            (design_id, simulation_id, critique)
        )
        return cursor.lastrowid

def get_design(design_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a design by ID."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM designs WHERE id = ?", (design_id,))
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "name": row[1],
                "parameters": json.loads(row[2]),
                "graph_blob": row[3],
                "created_at": row[4]
            }
        return None

def get_simulation(simulation_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a simulation by ID."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM simulations WHERE id = ?", (simulation_id,))
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "design_id": row[1],
                "mesh_path": row[2],
                "solver_path": row[3],
                "mean_cov": row[4],
                "pressure_drop": row[5],
                "objective": row[6],
                "status": row[7],
                "created_at": row[8]
            }
        return None