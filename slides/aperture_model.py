# /// script
# requires-python = ">=3.11"
# dependencies = ["xsuite", "numpy", "matplotlib", "xplt", "pint"]
# ///


# %% Toy ring survey and optics [slide]
# %% [markdown]
#
# Points of interest and survey references:
#
# - Six 60-degree arcs contain the dipoles (`mb.*`) and FODO quadrupoles
#   (`qf.*`, `qd.*`), used to anchor the magnet pipes and their transitions.
# - `chicane.start` and `chicane.end` delimit the four-bend chicane; the aggregate
#   chicane chamber is installed from `chicane.start`.
# - `ip` marks the centre of the widened chamber inside the chicane.
# - `short.mid.*` and `utility.mid.*` mark the centres of the straight sections,
#   where the straight pipes are installed; `rf.*` identifies the RF cavities.


# %%
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

env = xt.load("toy_ring.json")
ring = env["ring"]
print(ring)
# %% Survey [slide]
# %% [markdown]
#
# Labels identify the aperture survey references.


# %%
survey = ring.survey()
survey.plot(
    labels={
        r"chicane\.(start|end)|ip": {"width": 2.6},
        r"(short|utility)\.mid\..*|rf\..*": {},
        r"mb\..*": {"width": 0.8, "fontsize": 9},
        r"qf\..*": {"width": 1.7, "fontsize": 9},
        r"qd\..*": {"width": 2.6, "fontsize": 9},
    },
)
plt.gcf().set_size_inches(8, 8)
plt.title("Toy ring survey and aperture survey references")
# %% Optics [slide]
# %% [markdown]
#
# Beta functions and dispersion around the ring.


# %%
twiss = ring.twiss()
twiss.plot(figsize=(12, 5))
plt.title(f"Toy ring optics (qx = {twiss.qx:.3f}, qy = {twiss.qy:.3f})")
# %% Aperture toy ring [slide]
# %% [markdown]
#
# We now build an aperture model for the same line.
#
# The important modelling choice is that the aperture model is not forced to have one pipe per
# lattice element. Instead we use a small set of reusable pipe descriptions:
#
# - curved elliptical pipes for the arc bends;
# - octagonal straight pipes for the quadrupoles;
# - circular straight pipes for ordinary drifts;
# - one aggregate chicane chamber installed from `chicane.start` to `chicane.end`;
# - short transition pipes, between neighbouring main pipes.
#
# The construction below is deliberately split in two phases. First we build and place the main
# pipes. Then we inspect the gaps between those main pipes and place the required transition
# pipes. This keeps the aperture model tied to physical neighbours rather than to automatically
# generated drift names.


# %% [markdown]
# Profiles and pipes belong to the environment, while their placements and the built aperture
# model belong to the line. We can define the geometry incrementally and build the model once all
# pipes are placed.


# %%
print(env.aperture)
# %%
model = ring.aperture
print(model)
# %% Profiles [slide]
# %% [markdown]
# Profiles are transverse cross sections. Let's create the profiles used by the current model:
# `arc_ellipse` for the dipole chambers, `quad_octagon` for quadrupoles,
# and `drift_circle` for ordinary drift space.


# %%
arc_ellipse = env.aperture.new_profile(
    "arc_ellipse",
    "Ellipse",
    a=0.055,
    b=0.034,
)

fig, ax = plt.subplots(figsize=(3.5, 2.2))
arc_ellipse.plot(ax=ax)
fig.tight_layout()
# %%
quad_octagon = env.aperture.new_profile(
    "quad_octagon",
    "Octagon",
    max_x=0.048,
    max_y=0.048,
    max_diagonal=0.050,
)

fig, ax = plt.subplots(figsize=(3.5, 2.2))
quad_octagon.plot(ax=ax)
fig.tight_layout()
# %%
drift_circle = env.aperture.new_profile(
    "drift_circle",
    "Circle",
    radius=0.040,
)

fig, ax = plt.subplots(figsize=(3.5, 2.2))
drift_circle.plot(ax=ax)
fig.tight_layout()
# %% Profiles for the widened chicane insertion [slide]
# %% [markdown]
# `ip_entry` and `ip_chamber` are the profiles for the widened chicane insertion.


# %%
ip_entry = env.aperture.new_profile(
    "ip_entry",
    "RectEllipse",
    max_x=0.110,
    max_y=0.045,
    a=0.125,
    b=0.055,
)

fig, ax = plt.subplots(figsize=(3.5, 2.2))
ip_entry.plot(ax=ax)
fig.tight_layout()
# %%
ip_chamber = env.aperture.new_profile(
    "ip_chamber",
    "Racetrack",
    max_x=0.270,
    max_y=0.055,
    a=0.012,
    b=0.010,
)

fig, ax = plt.subplots(figsize=(3.5, 2.2))
ip_chamber.plot(ax=ax)
fig.tight_layout()
# %% Inspect a profile [slide]
# %% [markdown]
# The same profiles are accessible through `ring.aperture.profiles`. We can list their names,
# inspect a shape's parameters, or obtain the polygon used to draw it:


# %%
print(ring.aperture.profiles.keys())
profile = ring.aperture.profiles["arc_ellipse"]
print(profile.shape)
profile_polygon = profile.build_polygon(len_points=128)
print(profile_polygon[:5])
# %% Main pipes: arc bend pipe [slide]
# %% [markdown]
#
# A pipe is a longitudinal sequence of installed profiles. The same pipe can later be installed
# many times at different survey references.
#
#
# The bend pipe follows the bending radius of the `mb` element and uses the same elliptical
# profile at both ends.


# %%
l_mb = env["mb"].length
ang_mb = env["mb"].angle

arc_bend_pipe = env.aperture.new_pipe(
    "arc_bend_pipe",
    curvature=ang_mb / l_mb,
    positions=[
        arc_ellipse.place(shift_s=0.0),
        arc_ellipse.place(shift_s=l_mb),
    ],
)
arc_bend_pipe.plot()
# %% Inspect the bend pipe [slide]
# %% [markdown]
# To inspect the pipe behind these plots, use `ring.aperture.pipes`. A pipe can be iterated
# or indexed to access its profile positions and their properties:


# %%
pipe = ring.aperture.pipes["arc_bend_pipe"]
print(f"Length: {pipe.length:.3f} m; curvature: {pipe.curvature:.3f} 1/m")
print([(pos.profile.name, pos.shift_s) for pos in pipe])
# %%
arc_bend_pipe.plot_3d(frame="curved");
# %% Quadrupole pipe [slide]
# %% [markdown]
#
# The quadrupole chamber is straight. We rotate the octagon slightly around the local pipe axis to
# make it visually distinct from an axis-aligned octagon.


# %%
l_mq = float(env["l_mq"])

quad_pipe = env.aperture.new_pipe(
    "quad_pipe",
    positions=[
        # we can specify transformations, e.g. rot_s_rad to rotate around local s
        quad_octagon.place(shift_s=0.0, rot_s_rad=np.deg2rad(22.5)),
        quad_octagon.place(shift_s=l_mq, rot_s_rad=np.deg2rad(22.5)),
    ],
)
quad_pipe.plot_3d()
# %% Chicane pipe [slide]
# %% [markdown]
#
# The chicane is modelled as one aggregate chamber rather than four separate bend chambers. The
# lattice bends the reference trajectory outwards and back, meanwhile the aperture pipe is
# straight and installed from `chicane.start`.


# %%
chicane_z0 = float(survey.rows["chicane.start"].Z[0])
chicane_z1 = float(survey.rows["chicane.end"].Z[0])
ip_z = float(survey.rows["ip"].Z[0])

chicane_length = chicane_z1 - chicane_z0
ip_s = ip_z - chicane_z0
chamber_width = 5.0


print(f"Chicane length: {chicane_length:.3f} m")
print(f"IP position in the pipe: {ip_s:.3f} m")
print(f"Widened chamber length: {chamber_width:.3f} m")
# %% Construct the chicane pipe [slide]
# %%
chicane_pipe = env.aperture.new_pipe(
    "chicane_pipe",
    positions=[
        # Edge profiles are circles (smooth connection to neighbouring pipes)
        drift_circle.place(at=0.0),
        # Widen a bit 20 cm from the edges by installing `ip_entry`
        ip_entry.place(at=0.2),
        # Around the "IP" widen with `ip_chamber` and shift outwards
        ip_chamber.place(at=ip_s - chamber_width / 2, shift_x=0.15),
        ip_chamber.place(at=ip_s + chamber_width / 2, shift_x=0.15),
        ip_entry.place(at=chicane_length - 0.2),
        drift_circle.place(at=chicane_length),
    ],
)

chicane_pipe.plot(plane="zx")
plt.ylim(-1, 1)
# %%
chicane_pipe.plot_3d();
# %% Straight-section drift pipe [slide]
# %% [markdown]
#
# The straight insertions between arcs use a circular drift pipe. We define the pipe around its
# centre (`shift_s=-s` and `shift_s=+s`) so that it can be installed directly at the midpoint
# markers `short.mid.*` and `utility.mid.*`.


# %%
sector_straight_length = chicane_length
straight_half_length = 0.5 * sector_straight_length - 1e-6

straight_pipe = env.aperture.new_pipe(
    "straight_pipe",
    positions=[
        drift_circle.place(at=-straight_half_length),
        drift_circle.place(at=straight_half_length),
    ],
)
straight_pipe.plot()
plt.ylim(-1, 1)
# %% Install the main pipes [slide]
# %% [markdown]
#
# We now place all the main pipes.
#
# At this step, there are some gaps in the model, which we will address afterwards by constructing
# another set of transition pipes.


# %%
mb_names = survey.rows[r'mb\..*'].name
quad_names = survey.rows[r'q[fd]\..*'].name
straight_names = survey.rows[r'(short.mid|utility.mid)\..*'].name

ring.aperture.place_pipe("chicane_pipe", chicane_pipe, at="chicane.start")

ring.aperture.place_pipe("arc_bend", arc_bend_pipe, at=mb_names)
ring.aperture.place_pipe("quad", quad_pipe, at=quad_names)
ring.aperture.place_pipe("straight", straight_pipe, at=straight_names)

print(model)
# %% Transition pipes [slide]
# %% [markdown]
#
# **Currently, our model has gaps.** This is not, strictly speaking, a problem for us, as aperture
# computation routines will automatically interpolate the gaps. This happens along a straight
# line, and we do not have any gaps on arc sections. However, for completeness of the model, we
# will now construct transition pipes for our model.
#
#
# After the main pipes are installed, the remaining aperture pieces are short transitions between
# neighbouring main pipes.


# %%
survey.plot(labels=[r'q[df]\..*', r'mb\..*'])
plt.gcf().set_figwidth(6)
plt.gcf().set_figheight(6)
# %% Identify the required transitions [slide]
# %% [markdown]
# Zooming into one cell of the arc, we can identify what transition pipes are needed:


# %%
model.plot_floor_projection(s_range=(8, 12), aspect='equal')
env.aperture.discard_model()
# %%
cell_sv = env["arc_cell"].survey()
cell_sv.cols["name", "s", "length"].show()
# %% [markdown]
# In the survey we can see that all the arcs, and all the straight sections (not counting the
# chicane) are the same.
#
# Since we have placed pipes at all of the magnets, and at the chicane, the remaning parts of the
# aperture model that we still need to cover are the transitions between:
# - chicane end/long straight section to `qf` of length 0.19 m
# - `qf` and `mb` of length 0.238 m
# - `mb` and `qd` of length 0.288 m (works in both directions)
# - `mb` and the drift pipe
#
# We will also need to construct a suitable pipe for the short straight section in the middle of
# the arc.


# %% Measure the transitions [slide]
# %% [markdown]
# Let's collect some values:


# %%
edge_to_qf_len = cell_sv.rows[0].length[0]
qf_to_mb_len = cell_sv.rows['qf>>1'].length[0]
mb_to_qd_len = cell_sv.rows['mb::0>>1'].length[0]
qd_to_mb_len = cell_sv.rows['qd>>1'].length[0]
mb_to_edge_len = cell_sv.rows['mb::1>>1'].length[0]
mb_to_drift_len = edge_to_qf_len
edge_drift_len = mb_to_edge_len - mb_to_drift_len

# assert that we can reuse mb_to_qd pipe but "flipped"
assert np.isclose(mb_to_qd_len, qd_to_mb_len, rtol=0, atol=1e-6)
assert edge_drift_len > 0

print(f'- transition pipe from cell/straight edge to qf: {edge_to_qf_len} m')
print(f'- transition pipe from qf to mb: {qf_to_mb_len} m')
print(f'- transition pipe from mb to qd and qd to mb: {mb_to_qd_len} m')
print(f'- interface from mb to cell/straight edge: {mb_to_edge_len} m')
print(f'  - first {mb_to_drift_len} m transition from bend ellipse to drift circle')
print(f'  - remaining {edge_drift_len} m as circular drift pipe')
# %% Construct the transition pipes: edge → qf → mb [slide]
# %%
edge_qf_trans = env.aperture.new_pipe(
    'edge_qf_trans',
    positions=[
        drift_circle.place(at=0),
        quad_octagon.place(at=edge_to_qf_len),
    ],
)
edge_qf_trans.plot_3d()
# %%
qf_to_mb_trans = env.aperture.new_pipe(
    'qf_to_mb_trans',
    positions=[
        quad_octagon.place(at=0),
        arc_ellipse.place(at=qf_to_mb_len),
    ],
)
qf_to_mb_trans.plot_3d()
# %% Construct the transition pipes: mb → qd and mb → edge [slide]
# %%
mb_to_qd_trans = env.aperture.new_pipe(
    'mb_to_qd_trans',
    positions=[
        arc_ellipse.place(at=0),
        quad_octagon.place(at=mb_to_qd_len, rot_s_rad=np.deg2rad(22.5)),
    ],
)
mb_to_qd_trans.plot_3d()
# %%
mb_to_edge_iface = env.aperture.new_pipe(
    'mb_to_edge_iface',
    positions=[
        arc_ellipse.place(at=0),
        drift_circle.place(at=0.19),
        drift_circle.place(at=mb_to_edge_len),
    ],
)
mb_to_edge_iface.plot_3d()
# %% Install the transition pipes [slide]
# %% [markdown]
#
# Now that the few transition pipes exist, installing them is mechanical. We use the physical
# neighbours as anchors.


# %%
qf_names = list(survey.rows[r'qf\..*'].name)
qd_names = list(survey.rows[r'qd\..*'].name)
mb_names = list(survey.rows[r'mb\..*'].name)

mb_to_qd_names = mb_names[0::2]
qd_to_mb_names = mb_names[1::2]

print("Focusing quadrupoles:", qf_names)
print("Defocusing quadrupoles:", qd_names)
print("Bends before qd:", mb_to_qd_names)
print("Bends after qd:", qd_to_mb_names)
# %%
# `edge_qf_trans` ends at the start of each `qf`, so we place it at `qf@start`
ring.aperture.place_pipe(
    'edge_qf_trans', edge_qf_trans,
    at=[f'{name}@start' for name in qf_names],
    shift_z=-edge_to_qf_len,
)
# `qf_to_mb_trans` starts at the end of `qf`
ring.aperture.place_pipe(
    'qf_to_mb_trans', qf_to_mb_trans,
    at=[f'{name}@end' for name in qf_names],
)

print(model)
# %%
# `mb_to_qd_trans` starts at the end of `mb`
ring.aperture.place_pipe(
    'mb_to_qd_trans', mb_to_qd_trans,
    at=[f'{name}@end' for name in mb_to_qd_names],
)
# `qd_to_mb_trans` reused backwards so we place it at `mb@start` and rotate by 180° around y
ring.aperture.place_pipe(
    'qd_to_mb_trans', mb_to_qd_trans,
    at=[f'{name}@start' for name in qd_to_mb_names],
    rot_y_rad=np.pi,
)
# `mb_to_edge_iface` starts at the end of the second bend in each cell
# and carries the pipe to the next cell edge/straight section/chicane
ring.aperture.place_pipe(
    'mb_to_edge_iface', mb_to_edge_iface,
    at=[f'{name}@end' for name in qd_to_mb_names],
)
print(model)
# %% Build and inspect the aperture model [slide]
# %% [markdown]
#
# All profiles and pipes are registered in the environment, and all placements belong to
# `ring.aperture`. Building freezes the placement survey and prepares derived geometry for
# computation.


# %%
print("num_profile_points =", model.num_profile_points)
print("s_tol =", model.s_tol)
model.build_model();
print(model)
# %% The pipe table [slide]
# %% [markdown]
#
# The most basic way to inspect the model is to look at the pipe table, which lists all of the
# pipes and their locations:


# %%
pipe_table = model.get_pipe_table()
pipe_table.rows[:12].show()
# %% [markdown]
# The columns of the pipe table are:


# %%
print(pipe_table.cols)
# %% Pipe positions and spans [slide]
# %% [markdown]
# In particular:
# - `s_start` and `s_end` refer to the placement of the most extreme profiles of the pipe along
#   the survey, where the profile's $s$-position is calculated as the intersection of the profile
#   plane with the survey curve
# - `s_span_start` and `s_span_end` refer, similarly, to the most extreme $s$-span positions of
#   the pipe's profiles, where this is calculated by looking at all profile polygons placed in the
#   survey frame, projecting their points onto the survey curve, and taking the most extreme
#   values of $s$ obtained this way
# - `length` refers simply to `s_end - s_start`, while `span` refers to `s_span_end -
#   s_span_start`
#
# If all the profiles are installed orthogonally to the survey, the `s_span_...` and `s_...`
# quantities are equal.
#
# The pipe table is a **usual Xsuite table**, meaning all the same operations as on the twiss or
# survey table are supported.
# E.g. to get the information about the pipe installed w.r.t. `mb.5`:


# %%
pt = ring.aperture.get_pipe_table()
pt.rows[pt.survey_reference == 'mb.5'].show()
# %% The aperture bounds table [slide]
# %% [markdown]
#
# We can investigate the positioning of all the profiles of all the pipes using the bounds table:


# %%
bounds_table = model.get_bounds_table()
bounds_table.rows[:12].show()
print(bounds_table.cols)
# %% Profile positions and spans [slide]
# %% [markdown]
# Here it's the `s` values that determine the `s_start` and `s_end` of the pipe table, whereas
# `s_start` and `s_end` values determine the span.
#
#
#
# We can see that for the chicane pipe, the first two and the last two profiles have the same `s`,
# `s_start`, and `s_end`, and that this is true for most of the other profiles installed in the
# model.
#
# However, the 3rd and 4th profiles of the chicane pipe lie inside the chicane bends, and so they
# are not installed orthogonally to the survey: this is reflected in the fact that `s`, `s_start`,
# and `s_end` values differ.


# %%
fig, ax = plt.subplots(figsize=(6, 3))
ax.imshow(plt.imread("slides_assets/bounds.png"))
ax.axis("off")
fig.tight_layout()
# %%
bounds_table.rows[:6].show()
# %% Floor plot [slide]
# %% [markdown]
#
# The most basic plot is the floor plot of the aperture model. Below we overlay the survey on top
# as well:


# %%
ax = model.plot_floor_projection(aspect='equal')
ax.figure.set_figwidth(6)
ax.figure.set_figheight(6)
# %% Floor plot: a smaller chunk of the model [slide]
# %% [markdown]
# We can also choose to only plot a smaller chunk of the model, specifying `origin` and `s_range`:


# %%
ax = model.plot_floor_projection(origin=3.9, s_range=(-5, 5))
# ax = model.plot_floor_projection(origin='edge_qf_trans.3', s_range=(-3, 3))
ax.set_aspect('equal')
ax.set_ylim(-1, 1)
plt.gcf().set_figwidth(7)
# %% Follow the pipe placements [slide]
# %% [markdown]
# Each "box" in the floor plot corresponds to a placement in `ring.aperture.pipe_positions`.
# These placements refer to reusable pipes in `ring.aperture.pipes`, which in turn contain
# positions of profiles from `ring.aperture.profiles`. We can follow that chain directly:


# %%
print(ring.aperture.pipe_positions.keys()[:5])
placement = ring.aperture.pipe_positions["chicane_pipe"]
print(placement)
# %%
# Iterate over individual profile positions
for pos in placement.pipe:
    profile = ring.aperture.profiles[pos.profile.name]
    print(pos)
    print("    => has shape ", pos.profile.shape)
# %% Extents plot [slide]
# %% [markdown]
#
# The second useful type of plot is seeing the aperture limitation from the point of view of the
# beam.
#
# For this we can use the `plot_extents` function, which requires a parameter `s_positions`: the
# $s$ positions at which we will sample the model to compute the cross section of the aperture
# model seen by the beam.
#
# The convenience function `get_s_around_transitions` allows us to generate a list of $s$
# positions with a given resolution, but also adding $s$ positions where the geometry of the model
# might change. This way, we can capture many of the points of interest of the model.
#
# The extents plot shows the beam envelope at the maximum permissible size, but this can be
# tweaked using the `sigmas` parameter. The dashed line on the plot includes the aperture
# tolerances specified in the profile positions (and interpolated where necessary).


# %%
s_sample = model.get_s_around_transitions(resolution=0.1)
model.plot_extents(s_positions=s_sample)
plt.gcf().set_figwidth(8);
# %% Extents plot: 1 sigma between 20 and 40 m [slide]
# %% [markdown]
# For example, we can see the beam at $1\sigma$ between $s = 20\,\text{m}$ and $s = 40\,\text{m}$
# like so:


# %%
s_sample = model.get_s_around_transitions(resolution=0.1, s_range=(20, 40))
model.plot_extents(s_positions=s_sample, sigmas=1)
plt.gcf().set_figwidth(8);
# %% Save the aperture data [slide]
# %% [markdown]
#
# `env.aperture.to_json` saves the shared profiles and pipes, together with the placements
# and settings of every named line in the environment (including `ring`). It saves the
# aperture data separately from the lattice in `toy_ring.json`.
#
# Alternatively, `env.to_json("toy_ring_with_aperture.json", include_aperture=True)` saves the
# lattice and aperture data together.


# %%
env.aperture.to_json("aperture_model.json")
print("Saved aperture_model.json")
