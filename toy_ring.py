"""Six-arc aperture-model toy ring.

This example is deliberately small but geometrically rich.  The lattice is a
single-beam six-sector ring with 60 degree arcs and one insertion containing a
horizontal four-bend chicane.  The aperture model is built independently from
the magnetic lattice so that documentation can show survey references, curved
pipes, profile interpolation, local transforms, limit-element generation, and
N1 calculations on the same machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import xtrack as xt
from xtrack.aperture import Aperture, ApertureBuilder


@dataclass
class ToyRing:
    env: xt.Environment
    line: xt.Line
    aperture: Aperture
    twiss: xt.TwissTable


def build_line() -> tuple[xt.Environment, xt.Line]:
    """Build the six-sector toy ring and assign stable optics strengths."""
    env = xt.Environment()
    env.particle_ref = xt.Particles(p0c=5e9, mass0=xt.PROTON_MASS_EV)
    env.vars.default_to_zero = True

    n_arcs = 6
    cells_per_arc = 2
    bends_per_cell = 2
    n_arc_bends = n_arcs * cells_per_arc * bends_per_cell

    env["l_mb"] = 1.20
    env["l_mq"] = 0.32
    env["l_chicane_bend"] = 0.55
    env["ang_mb"] = 2 * np.pi / n_arc_bends
    env["ang_chicane"] = -np.deg2rad(9.0)

    env.new("mb", xt.RBend, length_straight="l_mb", angle="ang_mb")
    env.new("mq", xt.Quadrupole, length="l_mq")
    env.new("qf", "mq", k1="kqf")
    env.new("qd", "mq", k1="kqd")
    env.new("bc_p", xt.RBend, length_straight="l_chicane_bend", angle="ang_chicane")
    env.new("bc_m", xt.RBend, length_straight="l_chicane_bend", angle="-ang_chicane")
    env.new("rf", xt.Cavity, voltage=0.0, frequency=400e6)

    cell = env.new_line(
        name="arc_cell",
        length=4.80,
        components=[
            env.place("qf", at=0.35),
            env.place("mb", at=1.35),
            env.place("qd", at=2.40),
            env.place("mb", at=3.45),
        ],
    )
    arc = cell + cell

    chicane = env.new_line(
        name="chicane",
        length=7.80,
        components=[
            env.new("chicane.start", xt.Marker, at=0.00),
            env.place("bc_p", at=0.80),
            env.place("bc_m", at=2.15),
            env.new("ip", xt.Marker, at=3.90),
            env.place("bc_m", at=5.65),
            env.place("bc_p", at=7.00),
            env.new("chicane.end", xt.Marker, at=7.80),
        ],
    )
    chicane_survey = chicane.survey()
    sector_straight_length = float(chicane_survey.Z[-1] - chicane_survey.Z[0])

    utility_straight = env.new_line(
        name="utility_straight",
        length=sector_straight_length,
        components=[
            env.new("utility.mid", xt.Marker, at=sector_straight_length / 2),
            env.place("rf", at=2.45),
        ],
    )

    short_straight = env.new_line(
        name="short_straight",
        length=sector_straight_length,
        components=[
            env.new("short.mid", xt.Marker, at=sector_straight_length / 2)
        ],
    )

    ring = (
        chicane
        + arc
        + short_straight
        + arc
        + utility_straight
        + arc
        + short_straight
        + arc
        + utility_straight
        + arc
        + short_straight
        + arc
    )
    ring.replace_all_repeated_elements()
    ring.particle_ref = env.particle_ref
    ring.twiss_default["method"] = "4d"

    env.vars.update(
        {
            "kqf": 0.38,
            "kqd": -0.42,
        }
    )
    env.vars.default_to_zero = False
    env["ring"] = ring
    return env, ring


def build_aperture(line: xt.Line) -> Aperture:
    """Build a rich aperture model for the toy ring."""
    builder = ApertureBuilder(line)

    builder.new_profile("arc_ellipse", "Ellipse", half_major=0.055, half_minor=0.034)
    builder.new_profile(
        "quad_octagon",
        "Octagon",
        half_width=0.048,
        half_height=0.048,
        half_diagonal=0.050,
    )
    builder.new_profile("drift_circle", "Circle", radius=0.040)
    builder.new_profile(
        "ip_entry",
        "RectEllipse",
        half_width=0.110,
        half_height=0.045,
        half_major=0.125,
        half_minor=0.055,
    )
    builder.new_profile(
        "ip_chamber",
        "Racetrack",
        half_width=0.270,
        half_height=0.055,
        half_major=0.012,
        half_minor=0.010,
        tol_x=0.0015,
        tol_y=0.001,
    )

    l_mb = float(line.env["mb"].length)
    l_mq = float(line.env["l_mq"])
    ang_mb = float(line.env["ang_mb"])

    arc_bend_pipe = builder.new_pipe(
        "arc_bend_pipe",
        curvature=ang_mb / l_mb,
        positions=[
            builder.place_profile("arc_ellipse", shift_s=0.0),
            builder.place_profile("arc_ellipse", shift_s=l_mb),
        ],
    )
    quad_pipe = builder.new_pipe(
        "quad_pipe",
        positions=[
            builder.place_profile(
                "quad_octagon", shift_s=0.0, rot_s_rad=np.deg2rad(4.0)
            ),
            builder.place_profile(
                "quad_octagon", shift_s=l_mq, rot_s_rad=np.deg2rad(4.0)
            ),
        ],
    )
    table = line.get_table()
    chicane_s0 = float(table.rows["chicane.start"].s[0])
    chicane_s1 = float(table.rows["chicane.end"].s[0])
    survey = line.survey()
    chicane_z0 = float(survey.rows["chicane.start"].Z[0])
    chicane_z1 = float(survey.rows["chicane.end"].Z[0])
    ip_z = float(survey.rows["ip"].Z[0])

    chicane_pipe_length = chicane_z1 - chicane_z0
    ip_s = ip_z - chicane_z0
    chamber_width = 5.0
    chicane_pipe = builder.new_pipe(
        "chicane_pipe",
        positions=[
            builder.place_profile("drift_circle", shift_s=0.0),
            builder.place_profile("ip_entry", shift_s=0.2),
            builder.place_profile(
                "ip_chamber",
                shift_s=ip_s - chamber_width / 2,
                shift_x=0.15,
            ),
            builder.place_profile(
                "ip_chamber",
                shift_s=ip_s + chamber_width / 2,
                shift_x=0.15,
            ),
            builder.place_profile("ip_entry", shift_s=chicane_pipe_length - 0.2),
            builder.place_profile("drift_circle", shift_s=chicane_pipe_length),
        ],
    )

    sector_straight_length = chicane_pipe_length
    straight_half_length = 0.5 * sector_straight_length - 1e-6
    straight_pipe = builder.new_pipe(
        "straight_pipe",
        positions=[
            builder.place_profile("drift_circle", shift_s=-straight_half_length),
            builder.place_profile("drift_circle", shift_s=straight_half_length),
        ],
    )

    cell_sv = line.env["arc_cell"].survey()
    edge_to_qf_len = float(cell_sv.rows[0].length[0])
    qf_to_mb_len = float(cell_sv.rows["qf>>1"].length[0])
    mb_to_qd_len = float(cell_sv.rows["mb::0>>1"].length[0])
    qd_to_mb_len = float(cell_sv.rows["qd>>1"].length[0])
    mb_to_edge_len = float(cell_sv.rows["mb::1>>1"].length[0])
    mb_to_drift_len = edge_to_qf_len
    edge_drift_len = mb_to_edge_len - mb_to_drift_len

    assert np.isclose(mb_to_qd_len, qd_to_mb_len, rtol=0, atol=1e-6)
    assert edge_drift_len > 0

    edge_qf_bellow = builder.new_pipe(
        "edge_qf_bellow",
        positions=[
            builder.place_profile("drift_circle", shift_s=0.0),
            builder.place_profile("quad_octagon", shift_s=edge_to_qf_len),
        ],
    )
    qf_to_mb_bellow = builder.new_pipe(
        "qf_to_mb_bellow",
        positions=[
            builder.place_profile("quad_octagon", shift_s=0.0),
            builder.place_profile("arc_ellipse", shift_s=qf_to_mb_len),
        ],
    )
    mb_to_qd_bellow = builder.new_pipe(
        "mb_to_qd_bellow",
        positions=[
            builder.place_profile("arc_ellipse", shift_s=0.0),
            builder.place_profile("quad_octagon", shift_s=mb_to_qd_len),
        ],
    )
    mb_to_edge_iface = builder.new_pipe(
        "mb_to_edge_iface",
        positions=[
            builder.place_profile("arc_ellipse", shift_s=0.0),
            builder.place_profile("drift_circle", shift_s=mb_to_drift_len),
            builder.place_profile("drift_circle", shift_s=mb_to_edge_len),
        ],
    )

    mb_refs = [row.name for row in table.rows if row.name.startswith("mb.")]
    qf_refs = [row.name for row in table.rows if row.name.startswith("qf.")]
    qd_refs = [row.name for row in table.rows if row.name.startswith("qd.")]
    quad_refs = qf_refs + qd_refs
    straight_refs = [
        row.name for row in table.rows
        if row.name.startswith(("short.mid", "utility.mid"))
    ]

    mb_to_qd_refs = mb_refs[0::2]
    qd_to_mb_refs = mb_refs[1::2]
    assert len(qf_refs) == len(qd_refs) == len(mb_to_qd_refs) == len(qd_to_mb_refs)

    # One aggregate pipe spans the insertion. It is installed from the start
    # marker as a straight chamber rather than attached to each chicane bend.
    chicane_pipe.place("chicane_pipe", at="chicane.start")
    arc_bend_pipe.place("arc_bend", at=mb_refs)
    quad_pipe.place("quad", at=quad_refs)
    straight_pipe.place("straight", at=straight_refs)
    edge_qf_bellow.place(
        "edge_qf_bellow",
        at=[f"{name}@start" for name in qf_refs],
        shift_z=-edge_to_qf_len,
    )
    qf_to_mb_bellow.place(
        "qf_to_mb_bellow",
        at=[f"{name}@end" for name in qf_refs],
    )
    mb_to_qd_bellow.place(
        "mb_to_qd_bellow",
        at=[f"{name}@end" for name in mb_to_qd_refs],
    )
    mb_to_qd_bellow.place(
        "qd_to_mb_bellow",
        at=[f"{name}@start" for name in qd_to_mb_refs],
        rot_y_rad=np.pi,
    )
    mb_to_edge_iface.place(
        "mb_to_edge_iface",
        at=[f"{name}@end" for name in qd_to_mb_refs],
    )

    return Aperture(line=line, model=builder.build(), num_profile_points=128, s_tol=1e-5)


def build_toy_ring() -> ToyRing:
    """Build the lattice, optics table, and aperture model."""
    env, line = build_line()
    twiss = line.twiss()
    aperture = build_aperture(line)
    return ToyRing(env=env, line=line, aperture=aperture, twiss=twiss)


if __name__ == "__main__":
    toy = build_toy_ring()
    print(f"circumference = {toy.line.get_length():.6f} m")
    print(f"qx = {toy.twiss.qx:.6f}, qy = {toy.twiss.qy:.6f}")
    print(f"profiles = {list(toy.aperture.profiles.keys())}")
    print(
        toy.aperture.get_pipe_table().cols[
            "name pipe_name survey_reference s_start s_end length"
        ]
    )
