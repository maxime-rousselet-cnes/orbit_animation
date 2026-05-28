"""
Manim 3D scene showing a textured, rotating Earth with an orbiting satellite and its trace.
Observation ranges are visualized from randomly generated stations.
"""

from manimlib import *
from numpy import array, asin, clip, cos, dot, linspace, ndarray, ones_like, pi, sin, zeros_like
from numpy.linalg import norm
from numpy.random import random
from PIL.Image import open


def rotation_around_x_axis(angle: ndarray) -> ndarray:

    return array(
        object=[
            [ones_like(a=angle), zeros_like(a=angle), zeros_like(a=angle)],
            [zeros_like(a=angle), cos(angle), -sin(angle)],
            [zeros_like(a=angle), sin(angle), cos(angle)],
        ],
        dtype=float,
    ).transpose(
        2, 0, 1
    )  # (n_times, 3, 3).


def rotation_around_y_axis(angle: ndarray) -> ndarray:

    return array(
        object=[
            [cos(angle), zeros_like(a=angle), -sin(angle)],
            [zeros_like(a=angle), ones_like(a=angle), zeros_like(a=angle)],
            [sin(angle), zeros_like(a=angle), cos(angle)],
        ],
        dtype=float,
    ).transpose(
        2, 0, 1
    )  # (n_times, 3, 3).


def rotation_around_z_axis(angle: ndarray) -> ndarray:

    return array(
        object=[
            [cos(angle), -sin(angle), zeros_like(a=angle)],
            [sin(angle), cos(angle), zeros_like(a=angle)],
            [zeros_like(a=angle), zeros_like(a=angle), ones_like(a=angle)],
        ],
        dtype=float,
    ).transpose(
        2, 0, 1
    )  # (n_times, 3, 3).


class OrbitGeometricParameters:
    """
    Constant Geometric parameters of the satellite's orbit.
    """

    earth_radius: float
    semi_major_axis: float
    eccentricity: float
    inclination: float

    def __init__(
        self,
        earth_radius: float,
        semi_major_axis_factor: float,
        eccentricity: float,
        inclination: float,
    ) -> None:
        """
        Defines the constant geometric parameters of the satellite's orbit.
        """

        self.earth_radius = earth_radius
        self.semi_major_axis = semi_major_axis_factor * earth_radius
        self.eccentricity = eccentricity
        self.inclination = inclination

    def p_parameter(self) -> float:
        """
        Returns the p parameter usefull for circular coordinates.
        """

        return self.semi_major_axis * (1 - self.eccentricity**2)


class OrbitPartials:
    """
    All orbit partial derivatives needed for animation.
    """

    satellite_state_inertial: ndarray = array(object=[])
    partial_satellite_state_inertial_over_partial_i: ndarray = array(object=[])
    partial_satellite_state_inertial_over_partial_e: ndarray = array(object=[])
    partial_satellite_state_inertial_over_partial_initial_raan: ndarray = array(object=[])
    partial_satellite_state_inertial_over_partial_j_2: ndarray = array(object=[])
    partial_q_over_partial_i: ndarray = array(object=[])
    partial_q_over_partial_e: ndarray = array(object=[])
    partial_q_over_partial_initial_raan: ndarray = array(object=[])
    partial_q_over_partial_j_2: ndarray = array(object=[])


def is_visible(
    satellite_state_inertial: ndarray,
    station_coordinates: ndarray,
    earth_radius: float,
) -> bool:
    """
    Verifies if the satellite is visible from the station.
    """

    return dot(a=satellite_state_inertial, b=station_coordinates) > earth_radius**2


class OrbitModel:
    """
    Parameters of the satellite's orbit.
    """

    j_2: float
    period: float
    pulsation: float
    initial_raan: float
    orbit_geometric_parameters: OrbitGeometricParameters
    # Satellite inertial state per timestamp, per station.
    observation_models: dict[int, dict[str, float]] = {}
    observation_points: dict[int, dict[str, ndarray]] = {}
    orbit_partials: OrbitPartials = OrbitPartials()

    def __init__(
        self,
        j_2: float,
        period: float,
        initial_raan: float,
        orbit_geometric_parameters: OrbitGeometricParameters,
    ) -> None:
        """
        Defines the satellite's orbit model.
        """

        self.j_2 = j_2
        self.period = period
        self.pulsation = 2 * pi / self.period
        self.orbit_geometric_parameters = orbit_geometric_parameters
        self.initial_raan = initial_raan

    def generate_orbit_from_time(
        self,
        times: ndarray,
        station_dynamic_coordinates: dict[str, ndarray],  # (n_times, 3, 1) per element.
    ) -> None:
        """
        Generates a satellite orbit and the associated partials for animation.
        """

        mean_anomaly = self.pulsation * times  # (n_times).
        partial_nu_over_partial_e = 2 * sin(mean_anomaly)  # (n_times).
        nu = (
            mean_anomaly + self.orbit_geometric_parameters.eccentricity * partial_nu_over_partial_e
        )  # (n_times).
        cos_nu = cos(nu)  # (n_times).
        eccentric_ratio_factor = (
            1 + self.orbit_geometric_parameters.eccentricity * cos_nu
        )  # (n_times).
        r = self.orbit_geometric_parameters.p_parameter() / eccentric_ratio_factor  # (n_times).
        r_z_nu = rotation_around_z_axis(angle=nu)  # (n_times, 3, 3).
        circular_coordinates_in_orbital_plane = array(
            object=[[r], [zeros_like(a=r)], [zeros_like(a=r)]]
        ).transpose(
            2, 0, 1
        )  # (n_times, 3, 1).
        satellite_state_orbital_plane = (
            r_z_nu @ circular_coordinates_in_orbital_plane
        )  # (n_times, 3, 1).
        raan_temp = (
            -3
            / 2
            * self.pulsation
            * (
                self.orbit_geometric_parameters.earth_radius
                / self.orbit_geometric_parameters.semi_major_axis
            )
            ** 2
            / (1 - self.orbit_geometric_parameters.eccentricity**2) ** 2
        )  # float.
        partial_raan_dot_over_partial_j_2 = raan_temp * cos(
            self.orbit_geometric_parameters.inclination
        )  # float.
        raan_dot = partial_raan_dot_over_partial_j_2 * self.j_2  # float.
        raan = self.initial_raan + raan_dot * times  # (n_times).
        r_z_raan = rotation_around_z_axis(angle=raan)  # (n_times, 3, 3).
        r_x_i = rotation_around_x_axis(
            angle=array(object=[self.orbit_geometric_parameters.inclination])
        )  # (1, 3, 3)
        rotation_from_orbital_plane_to_inertial = r_z_raan @ r_x_i  # (n_times, 3, 3).
        self.orbit_partials.satellite_state_inertial = (
            rotation_from_orbital_plane_to_inertial @ satellite_state_orbital_plane
        )  # (n_times, 3, 1).
        partial_raan_dot_over_partial_e = (
            4
            * partial_raan_dot_over_partial_j_2
            * self.orbit_geometric_parameters.eccentricity
            / (1 - self.orbit_geometric_parameters.eccentricity**2)
        )  # float.
        partial_raan_dot_over_partial_i = -raan_temp * sin(
            self.orbit_geometric_parameters.inclination
        )  # float.
        r_z_quarter = rotation_around_z_axis(angle=array(object=[pi / 2]))  # (1, 3, 3).
        r_x_quarter = rotation_around_x_axis(angle=array(object=[pi / 2]))  # (1, 3, 3).
        self.orbit_partials.partial_satellite_state_inertial_over_partial_initial_raan = (
            r_z_quarter @ self.orbit_partials.satellite_state_inertial
        )  # (n_times, 3, 1).
        self.orbit_partials.partial_satellite_state_inertial_over_partial_i = (
            rotation_from_orbital_plane_to_inertial @ r_x_quarter @ satellite_state_orbital_plane
            + partial_raan_dot_over_partial_i
            * self.orbit_partials.partial_satellite_state_inertial_over_partial_initial_raan
        )  # (n_times, 3, 1).
        partial_r_over_partial_e = -r * cos_nu / eccentric_ratio_factor  # (n_times).
        partial_r_z_nu_over_partial_e = (
            partial_nu_over_partial_e[:, None, None] * r_z_nu @ r_z_quarter
        )  # (n_times, 3, 3).
        partial_satellite_state_orbital_plane_over_partial_e = (
            partial_r_z_nu_over_partial_e @ circular_coordinates_in_orbital_plane
            + r_z_nu
            @ array(
                object=[
                    [partial_r_over_partial_e],
                    [zeros_like(a=partial_r_over_partial_e)],
                    [zeros_like(a=partial_r_over_partial_e)],
                ]
            ).transpose(2, 0, 1)
        )  # (n_times, 3, 1).
        self.orbit_partials.partial_satellite_state_inertial_over_partial_e = (
            partial_raan_dot_over_partial_e
            * times[:, None, None]
            * self.orbit_partials.partial_satellite_state_inertial_over_partial_initial_raan
            + rotation_from_orbital_plane_to_inertial
            @ partial_satellite_state_orbital_plane_over_partial_e
        )  # (n_times, 3, 1).
        self.orbit_partials.partial_satellite_state_inertial_over_partial_j_2 = (
            self.orbit_partials.partial_satellite_state_inertial_over_partial_initial_raan
            * partial_raan_dot_over_partial_j_2
            * times[:, None, None]
        )  # (n_times, 3, 1).

        partial_q_over_partial_e = []
        partial_q_over_partial_i = []
        partial_q_over_partial_initial_raan = []
        partial_q_over_partial_j_2 = []

        for i_time, time in enumerate(times):

            for station_name, station_coordinates in station_dynamic_coordinates.items():

                if is_visible(
                    satellite_state_inertial=self.orbit_partials.satellite_state_inertial[
                        i_time, :, 0
                    ],
                    station_coordinates=station_coordinates[i_time, :, 0],
                    earth_radius=self.orbit_geometric_parameters.earth_radius,
                ):

                    if time not in self.observation_points:

                        self.observation_points[i_time] = {}
                        self.observation_models[i_time] = {}

                    self.observation_points[i_time][station_name] = (
                        self.orbit_partials.satellite_state_inertial[i_time]
                    )
                    observation_vector = (
                        self.orbit_partials.satellite_state_inertial[i_time]
                        - station_dynamic_coordinates[station_name][i_time]
                    )
                    self.observation_models[i_time][station_name] = norm(x=observation_vector)
                    observation_gradient = (
                        observation_vector / self.observation_models[i_time][station_name]
                    )
                    partial_q_over_partial_e += [
                        dot(
                            self.orbit_partials.partial_satellite_state_inertial_over_partial_e[
                                i_time, :, 0
                            ],
                            observation_gradient[:, 0],
                        )
                    ]
                    partial_q_over_partial_i += [
                        dot(
                            self.orbit_partials.partial_satellite_state_inertial_over_partial_i[
                                i_time, :, 0
                            ],
                            observation_gradient[:, 0],
                        )
                    ]
                    partial_q_over_partial_initial_raan += [
                        dot(
                            self.orbit_partials.partial_satellite_state_inertial_over_partial_initial_raan[
                                i_time, :, 0
                            ],
                            observation_gradient[:, 0],
                        )
                    ]
                    partial_q_over_partial_j_2 += [
                        dot(
                            self.orbit_partials.partial_satellite_state_inertial_over_partial_j_2[
                                i_time, :, 0
                            ],
                            observation_gradient[:, 0],
                        )
                    ]

        self.orbit_partials.partial_q_over_partial_e = array(object=partial_q_over_partial_e)
        self.orbit_partials.partial_q_over_partial_i = array(object=partial_q_over_partial_i)
        self.orbit_partials.partial_q_over_partial_initial_raan = array(
            object=partial_q_over_partial_initial_raan
        )
        self.orbit_partials.partial_q_over_partial_j_2 = array(object=partial_q_over_partial_j_2)


def build_ocean_mask(texture_path: str = "raster_images/earth_day.jpeg") -> np.ndarray:
    """
    Generates an ocan binary mask from the day texture image.
    """

    img = array(open(texture_path).convert("RGB"), dtype=float)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    is_ocean = (b > r + 10) & (b > g) & ((r + g + b) > 60) & ((r + g + b) < 700)

    return is_ocean


def sample_point_on_continents(mask: ndarray) -> tuple[float, float]:
    """
    Return (lat, lon) in radians of a uniformly random continent pixel.
    """

    while True:

        lat = asin(2 * random() - 1)
        lon = (2 * random() - 1) * pi
        px = int((lon / (2 * pi) + 0.5) * mask.shape[1]) % mask.shape[1]
        py = int((0.5 - lat / pi) * mask.shape[0]) % mask.shape[0]

        if not mask[py, px]:

            return lat, lon


def geographic_to_xyz(latitude: float, longitude: float, radius: float) -> np.ndarray:
    """
    Converts geographic coordinates (radians) to Cartesian on a sphere.
    """

    return radius * array(
        [
            [cos(latitude) * cos(longitude)],
            [cos(latitude) * sin(longitude)],
            [sin(latitude)],
        ]
    )


def generate_stations(
    times: ndarray,
    earth_rotation_rate: float,
    earth_radius: float,
    n_stations: int,
    mask: ndarray = build_ocean_mask(),
) -> dict[str, ndarray]:
    """
    Generates n_stations random stations on the continents.
    """

    station_dynamic_coordinates: dict[str, ndarray] = {}  # (n_times, 3, 1) per element.
    station_latitudes = []
    station_longitudes = []
    earth_rotation = rotation_around_z_axis(angle=earth_rotation_rate * times)  # (n_times, 3, 3).

    for i in range(n_stations):

        latitude, longitude = sample_point_on_continents(mask=mask)
        station_latitudes += [latitude]
        station_longitudes += [longitude]
        station_dynamic_coordinates[f"Station {i}"] = (
            earth_rotation
            @ geographic_to_xyz(latitude=latitude, longitude=longitude, radius=earth_radius)[
                None, :, :
            ]
        )  # (n_times, 3, 1)

    return station_dynamic_coordinates


class EarthAndSatelliteSceneGammaAnimated(ThreeDScene):
    """
    3D scene: textured rotating Earth, inclined satellite orbit, persistent trace,
    and periodic ground-station observation links.

    The full satellite trajectory and station positions are pre-computed on a dense
    time grid via OrbitModel / generate_stations before the scene starts.  During
    playback a single ValueTracker drives:
      - the satellite Sphere (index lookup into the pre-computed positions),
      - the orbital trace (a single VMobject whose points are ALL pre-computed
        positions up to the current index — fully modifiable at any time),
      - the observation links (green line + station dot shown whenever a
        pre-computed visibility window is active at the current frame).

    Nothing is ever lost in updaters: the trace lives entirely in one VMobject
    and the observation table is a plain dict built before animation starts.
    """

    earth_resolution: int = 100
    earth_radius: float = 2.2
    earth_rotation_period: float = 150.0  # (s).
    orbit_period: float = 40  # (s).
    semi_major_axis_factor: float = (6371 + 1500) / 6371
    eccentricity: float = 0.0
    inclination: float = 50.0 * DEGREES
    initial_raan: float = 0.0 * DEGREES
    j_2: float = -0.1  # Exaggerated for visual effect, not realistic!
    n_stations: int = 10
    frame_rate: int = 25  # fps.
    animation_duration: float = 80.0  # (s).
    day_texture: str = "earth_day.jpeg"
    night_texture: str = "earth_night.jpg"
    random_range_sigma: float = 0.05

    # Ramping J_2 values.
    n_0_j2 = 5.0  # Waits.
    n_1_j2 = 5.0  # Ramps and holds.

    delta_j_2 = 0.0
    delta_j_2_1 = 0.05
    delta_j_2_2 = -0.05

    def build_orbit_and_stations(
        self,
    ) -> tuple[np.ndarray, OrbitModel, dict[str, ndarray]]:
        """
        Pre-compute satellite the satellite's orbit, station positions, observations, observation
        points and partials for the whole time grid.
        """

        times = linspace(
            start=0,
            stop=self.animation_duration,
            num=int(self.animation_duration * self.frame_rate),
        )
        orbit_model = OrbitModel(
            j_2=self.j_2,
            period=self.orbit_period,
            initial_raan=self.initial_raan,
            orbit_geometric_parameters=OrbitGeometricParameters(
                earth_radius=self.earth_radius,
                semi_major_axis_factor=self.semi_major_axis_factor,
                eccentricity=self.eccentricity,
                inclination=self.inclination,
            ),
        )
        station_dynamic_coordinates = generate_stations(
            times=times,
            earth_rotation_rate=TAU / self.earth_rotation_period,
            earth_radius=self.earth_radius,
            n_stations=self.n_stations,
        )
        orbit_model.generate_orbit_from_time(
            times=times,
            station_dynamic_coordinates=station_dynamic_coordinates,
        )

        return times, orbit_model, station_dynamic_coordinates

    def current_index(self, n_steps: int, time_tracker: ValueTracker) -> int:
        """
        Map tracker value (0 … animation_duration) to a time-array index.
        """

        t = time_tracker.get_value()
        idx = int(t / self.animation_duration * (n_steps - 1))
        return max(0, min(idx, n_steps - 1))

    def construct(self) -> None:

        # Pre-computing.
        times, orbit_model, station_dynamic_coordinates = self.build_orbit_and_stations()
        n_steps = len(times)
        satellite_positions = orbit_model.orbit_partials.satellite_state_inertial[
            :, :, 0
        ]  # (n_steps, 3).

        # Earth rendering.
        earth_sphere = Sphere(
            radius=self.earth_radius,
            resolution=(2 * self.earth_resolution + 1, self.earth_resolution + 1),
            u_range=(0, TAU),
            v_range=(1e-3, PI - 1e-3),
        )
        earth = TexturedSurface(earth_sphere, self.day_texture, self.night_texture)
        earth.add_updater(lambda m, dt: m.rotate(TAU / self.earth_rotation_period * dt, axis=OUT))
        self.add(earth)

        # Camera & light.
        frame = self.camera.frame
        frame.set_euler_angles(theta=-45 * DEGREES, phi=60 * DEGREES)
        # self.camera.frame.shift(1.5 * RIGHT + IN)

        light = self.camera.light_source
        light.move_to(23.5 * RIGHT + 10 * OUT)

        # Time tracker.
        # One ValueTracker is the single source of truth for the current time.
        # All updaters derive their state from it so satellite, trace, and
        # observation links can never drift apart.
        time_tracker = ValueTracker(0.0)

        # Delta J_2 tracker.
        j2_tracker = ValueTracker(self.delta_j_2)
        tex = Tex(r"\Delta \gamma_i =", f"{self.delta_j_2:.3f}")
        j2_value = tex.make_number_changeable(f"{self.delta_j_2:.3f}")
        tex.to_edge(RIGHT)
        tex.fix_in_frame()
        self.add(tex)

        def update_j2_value(m):

            v = j2_tracker.get_value()
            m.set_value(v)

            if v >= 0 and self.delta_j_2_1 != 0:

                m.set_color(interpolate_color(WHITE, RED, v / self.delta_j_2_1))

            else:

                m.set_color(interpolate_color(WHITE, BLUE, v / self.delta_j_2_2))

        j2_value.add_updater(update_j2_value)

        def update_j2_tracker(m: ValueTracker):

            t1 = self.n_0_j2 + self.n_1_j2
            t2 = t1 + self.n_1_j2
            t3 = t2 + self.n_1_j2
            delta = self.delta_j_2
            t = time_tracker.get_value() % t3

            # Ramps up to delta_j_2_1.
            if self.n_0_j2 <= t < t1:

                a = smooth((t - self.n_0_j2) / self.n_1_j2)
                delta = interpolate(self.delta_j_2, self.delta_j_2_1, a)

            # Ramps down to delta_j_2_2.
            elif t1 <= t < t2:

                a = smooth((t - t1) / self.n_1_j2)
                delta = interpolate(self.delta_j_2_1, self.delta_j_2_2, a)

            # Ramps back to zero.
            elif t2 <= t < t3:

                a = smooth((t - t2) / self.n_1_j2)
                delta = interpolate(self.delta_j_2_2, 0.0, a)

            # Final hold.
            else:

                delta = 0.0

            m.set_value(delta)

        j2_tracker.add_updater(update_j2_tracker)
        self.add(j2_tracker)

        # Satellite.
        sat = Sphere(radius=0.06, resolution=(12, 6))
        sat.set_color(RED)
        sat.move_to(satellite_positions[0])
        self.add(sat)

        # Satellite updater: pure index lookup into pre-computed positions.
        def update_satellite(m: VMobject) -> None:
            """
            Satellite updater: index lookup into pre-computed positions.
            """
            idx = self.current_index(n_steps=n_steps, time_tracker=time_tracker)
            m.move_to(
                satellite_positions[idx]
                + orbit_model.orbit_partials.partial_satellite_state_inertial_over_partial_j_2[
                    idx, :, 0
                ]
                * j2_tracker.get_value()
            )

        sat.add_updater(update_satellite)

        # Orbital trace.
        # The trace is a single VMobject whose corner points are the full
        # pre-computed trajectory. The updater replaces the shown prefix each
        # frame — no data is ever appended or lost.
        all_trace_points = satellite_positions.copy()  # (n_steps, 3)
        trace = VMobject()
        trace.set_stroke(YELLOW, width=1.5, opacity=0.8)
        trace.set_points_as_corners(all_trace_points[:2])
        self.add(trace)

        # Trace updater: grow the visible prefix without ever losing points.
        def update_trace(m: VMobject) -> None:

            idx = self.current_index(n_steps=n_steps, time_tracker=time_tracker)

            if idx < 2:

                return

            m.set_points_as_corners(
                all_trace_points[: idx + 1]
                + orbit_model.orbit_partials.partial_satellite_state_inertial_over_partial_j_2[
                    : idx + 1, :, 0
                ]
                * j2_tracker.get_value()
            )

        trace.add_updater(update_trace)

        # Observations: green links between visible stations and the satellite.
        # A fixed pool of Line VMobjects is recycled each frame: lines beyond
        # the active count are made invisible so Manim never needs to add or
        # remove scene children during the animation.

        link_lines: dict[str, Line] = {}

        for station, station_positions in station_dynamic_coordinates.items():

            ln = Line(start=ORIGIN, end=station_positions[0, :, 0])
            ln.set_stroke(GREEN, width=1.2, opacity=0.7)
            self.add(ln)
            link_lines[station] = ln

        def update_links(_: VMobject) -> None:

            idx = self.current_index(n_steps=n_steps, time_tracker=time_tracker)

            for station, ln in link_lines.items():

                if (
                    idx in orbit_model.observation_points
                    and station in orbit_model.observation_points[idx]
                ):

                    ln.put_start_and_end_on(
                        start=orbit_model.observation_points[idx][station][:, 0],
                        end=station_dynamic_coordinates[station][idx, :, 0],
                    )

                else:

                    ln.put_start_and_end_on(
                        start=ORIGIN, end=station_dynamic_coordinates[station][idx, :, 0]
                    )

        # Attach the updater to the first line (or a dummy mobject) so Manim
        # calls it once per frame.  Using a VGroup avoids touching the lines
        # themselves with an extra updater.
        link_group = VGroup(*list(link_lines.values()))
        link_group.add_updater(update_links)
        self.add(link_group)

        observation_dots: dict[tuple[int, str], Sphere] = {}

        for idx, observation_points_per_station in orbit_model.observation_points.items():

            for station, observation_point in observation_points_per_station.items():

                dot = Sphere(radius=0.03, resolution=(12, 6))
                dot.set_color(GREEN, opacity=0.0)
                dot.move_to(observation_point[:, 0] + self.random_range_sigma * (random() - 0.5))
                self.add(dot)
                observation_dots[(idx, station)] = dot

        def update_dots(_: Mobject) -> None:

            idx = self.current_index(n_steps=n_steps, time_tracker=time_tracker)

            for (dot_idx, _), dot in observation_dots.items():

                if idx == dot_idx:

                    dot.set_opacity(0.7)

        dot_group = Group(*list(observation_dots.values()))
        dot_group.add_updater(update_dots)
        self.add(dot_group)

        self.play(
            time_tracker.animate.set_value(self.animation_duration),
            run_time=self.animation_duration,
            rate_func=linear,
        )
