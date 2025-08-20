"""
Servicio de integración con PyPSA.
Requiere: pypsa, numpy, pandas, scipy, networkx, (opcional) pyomo + glpk/cbc para LOPF.
"""
import json
from typing import Dict, Any, Optional

try:
    import pypsa  # type: ignore
except Exception as e:
    pypsa = None

class PyPSAUnavailable(Exception):
    pass

def ensure_pypsa():
    if pypsa is None:
        raise PyPSAUnavailable("PyPSA no está disponible en este entorno. Use Python>=3.8 y `pip install pypsa pyomo` (y un solver como glpk/cbc).")

def build_network_from_config(cfg: Dict[str, Any]):
    """
    Construye un pypsa.Network a partir de un dict:
    {
      "buses":[{"name":"BUS_A","v_nom":33.0},{"name":"BUS_B","v_nom":33.0}],
      "lines":[{"name":"L1","bus0":"BUS_A","bus1":"BUS_B","x_pu":0.1,"r_pu":0.02,"s_nom":20.0}],
      "transformers":[{"name":"T1","bus0":"BUS_A","bus1":"BUS_MT","s_nom":16.0,"x_pu":0.08,"r_pu":0.01,"tap_ratio":1.0}],
      "loads":[{"name":"LD1","bus":"BUS_B","p_mw":2.0,"q_mvar":0.5}],
      "generators":[{"name":"G1","bus":"BUS_A","p_set_mw":3.0,"v_set_pu":1.0,"p_max_mw":5.0}]
    }
    """
    ensure_pypsa()
    n = pypsa.Network()
    # Buses
    for b in cfg.get("buses", []):
        n.add("Bus", b["name"], v_nom=b.get("v_nom", 33.0))
    # Lines
    for ln in cfg.get("lines", []):
        n.add("Line", ln["name"],
              bus0=ln["bus0"], bus1=ln["bus1"],
              x_pu=ln.get("x_pu", 0.1), r_pu=ln.get("r_pu", 0.0),
              s_nom=ln.get("s_nom", 10.0))
    # Transformers
    for tr in cfg.get("transformers", []):
        n.add("Transformer", tr["name"],
              bus0=tr["bus0"], bus1=tr["bus1"],
              s_nom=tr.get("s_nom", 10.0),
              x_pu=tr.get("x_pu", 0.1), r_pu=tr.get("r_pu", 0.0),
              tap_ratio=tr.get("tap_ratio", 1.0))
    # Loads
    for ld in cfg.get("loads", []):
        n.add("Load", ld["name"], bus=ld["bus"],
              p_set=ld.get("p_mw", 0.0), q_set=ld.get("q_mvar", 0.0))
    # Generators (control de tensión con v_set)
    for g in cfg.get("generators", []):
        n.add("Generator", g["name"], bus=g["bus"],
              p_set=g.get("p_set_mw", 0.0),
              p_max_pu=1.0, p_nom=g.get("p_max_mw", 1.0),
              control="PV", v_set=g.get("v_set_pu", 1.0))
    return n

def run_power_flow(n, solver: Optional[str]=None) -> Dict[str, Any]:
    """
    Ejecuta flujo de potencia no lineal (AC) si está disponible.
    """
    ensure_pypsa()
    # AC power flow
    n.pf()  # calc voltages, flows
    out = {
        "bus_v_pu": n.buses_t.v_mag_pu.iloc[-1].to_dict() if not n.buses_t.v_mag_pu.empty else {},
        "line_p_mw": n.lines_t.p0.iloc[-1].to_dict() if not n.lines_t.p0.empty else {},
        "trafo_p_mw": n.transformers_t.p0.iloc[-1].to_dict() if not n.transformers_t.p0.empty else {},
        "gen_p_mw": n.generators_t.p.iloc[-1].to_dict() if not n.generators_t.p.empty else {},
        "load_p_mw": n.loads_t.p.iloc[-1].to_dict() if not n.loads_t.p.empty else {},
    }
    return out