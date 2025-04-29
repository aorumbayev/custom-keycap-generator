from build123d import *
import numpy as np
from dataclasses import dataclass
from stem import Stem
from build123d import Color, Mesher


@dataclass
class KeyConfig:
    tol: float
    tol_tight: float
    wall: float
    inner_rad: float

    key_h: float
    key_r: float

    back_slope: float
    front_slope: float
    side_slope: float
    back_curve: float
    front_curve: float

    front_dy: float
    back_dy: float
    width: float  # As a multiple of `key_h`
    bump: bool = False
    # Multi-legend support
    legend_center: str = ""
    legend_top_left: str = ""
    legend_top_right: str = ""

class Key:
    def __init__(self, config: KeyConfig, stem: Stem):
        self.tol = config.tol
        self.tol_tight = config.tol_tight
        self.wall = config.wall

        self.key_h = config.key_h
        self.key_w = self.key_h * config.width
        self.back_dy = config.back_dy
        self.front_dy = config.front_dy

        self.back_slope = np.radians(config.back_slope)
        self.front_slope = np.radians(config.front_slope)
        self.side_slope = np.radians(config.side_slope)

        self.back_curve = config.back_curve
        self.front_curve = config.front_curve

        self.max_back_height = self.back_dy + max(0.0, -self.back_curve) + 1.0
        self.max_front_height = self.front_dy + max(0.0, -self.front_curve) + 1.0
        self.max_height = max(self.max_back_height, self.max_front_height)

        self.key_r = config.key_r
        self.inner_rad = config.inner_rad
        self.eps = 0.001

        self.cross_height = 4.1 + self.tol
        self.cross_thick = 1.17 + self.tol_tight
        self.stem_depth = 3.8 + self.tol
        self.stem_rad = 0.3

        self.bump = config.bump
        # Store legends
        self.legend_center = config.legend_center
        self.legend_top_left = config.legend_top_left
        self.legend_top_right = config.legend_top_right

        self.stem = stem

    def _outer_key_profile(self, shift: float = 0.0) -> Part:
        """
        Generates the overall (non-hollow) outer shape of the key.
        This is handy for boolean operations. For example, subtracting a
        shifted profile from an un-shifted profile creates a hollow profile.

        :param shift: Distance the profile is shifted by
        """
        key_h = self.key_h + 2*shift
        key_w = self.key_w + 2*shift
        back_dy = self.back_dy + shift
        front_dy = self.front_dy + shift
        key_r = max(0.0, self.key_r + shift)

        """
        Start by constructing a trapezoidal prism.
        This is done via the intersection of two lofts.
        Each loft is between a pair of rectangular faces.
        Optionally, the front and back faces can have a curved top.
        """
        with BuildPart() as part:
            # First loft is between left and right sides (along x-axis)
            with BuildLine() as side_profile:
                Polyline(
                    (-key_h / 2 + np.tan(self.front_slope) * self.max_front_height, self.max_front_height),
                    (-key_h/2, 0),
                    (key_h/2, 0),
                    (key_h / 2 - np.tan(self.back_slope) * self.max_back_height, self.max_back_height),
                    close=True
                )
            with BuildSketch(Plane.YZ.offset(-key_w/2)):
                add(side_profile.line)
                make_face()
            with BuildSketch(Plane.YZ.offset(key_w/2)):
                add(side_profile.line)
                make_face()
            loft()

            # Project front/back heights onto un-sloped planes
            back_dx = np.tan(self.back_slope) * back_dy
            front_dx = np.tan(self.front_slope) * front_dy
            # Avoid division by zero for flat keys
            if abs(key_h - front_dx - back_dx) < self.eps:
                 top_slope = 0.0
            else:
                top_slope = (front_dy - back_dy) / (key_h - front_dx - back_dx)

            front_dy_proj = front_dy + top_slope * front_dx
            back_dy_proj = back_dy - top_slope * back_dx

            # Second loft is between front and back faces (along y-axis)
            with BuildSketch(Plane.XZ.offset(-key_h/2)):
                curved = abs(self.back_curve) > self.eps
                back_side_dx = np.tan(self.side_slope) * back_dy_proj
                with BuildLine():
                    Polyline(
                        (-key_w/2 + back_side_dx, back_dy_proj),
                        (-key_w/2, 0),
                        (key_w/2, 0),
                        (key_w/2 - back_side_dx, back_dy_proj),
                        close=not curved
                    )
                    if curved:
                        ThreePointArc(
                            (-key_w/2 + back_side_dx, back_dy_proj),
                            (0, back_dy_proj - self.back_curve),
                            (key_w/2 - back_side_dx, back_dy_proj)
                        )
                make_face()
            with BuildSketch(Plane.XZ.offset(key_h/2)):
                curved = abs(self.front_curve) > self.eps
                front_side_dx = np.tan(self.side_slope) * front_dy_proj
                with BuildLine():
                    Polyline(
                        (-key_w/2 + front_side_dx, front_dy_proj),
                        (-key_w/2, 0),
                        (key_w/2, 0),
                        (key_w/2 - front_side_dx, front_dy_proj),
                        close=not curved
                    )
                    if curved:
                        ThreePointArc(
                            (-key_w/2 + front_side_dx, front_dy_proj),
                            (0, front_dy_proj - self.front_curve),
                            (key_w/2 - front_side_dx, front_dy_proj)
                        )
                make_face()
            loft(mode=Mode.INTERSECT)

            # Finally, round off the edges
            if key_r > 0.0:
                edges = part.edges().group_by(Axis.Z)[1:]
                fillet(
                    sum(edges[1:], edges[0]),
                    key_r
                )
        return part.part

    def shape(self, return_components: bool = False) -> Part | tuple[Part, list[Part | None]]:
        """
        Constructs the key's complete shape or its components (body + list of legends).

        Args:
            return_components (bool, optional): If True, returns a tuple of
                                                (key_body, [legend_center, legend_tl, legend_tr]).
                                                Otherwise, returns the combined part.
                                                Defaults to False.

        Returns:
            Part | tuple[Part, list[Part | None]]: Combined shape or tuple of components.
        """

        # Construct hollow key shell via boolean operations
        # (avoid `offset` operation since it's completely unreliable)
        outer_profile = self._outer_key_profile()
        shell = outer_profile - self._outer_key_profile(shift=-self.wall)

        # Fill in the top of the key, so that stem is shorter, for strength
        with BuildPart() as top_block:
            with Locations((0.0, 0.0, self.stem_depth + self.inner_rad)):
                Box(
                    2*self.key_w, 2*self.key_h, self.max_height,
                    align=(Align.CENTER, Align.CENTER, Align.MIN)
                )
        filler = outer_profile & top_block.part

        # Add the stem
        cross = self.stem.build(self)
        # Combine base parts
        key_body = shell + filler + cross

        # Fillet some inside edges, for strength
        if self.inner_rad > 0.0:
            key_body = key_body.fillet(
                edge_list=self.stem.select_inner_rad_edges(self, key_body),
                radius=self.inner_rad - self.eps,
            )
        
        # Add bump if needed
        if self.bump:
            with BuildPart() as bump_part:
                y = self.stem_depth + self.inner_rad + self.eps
                with BuildSketch(Plane.XY.offset(y)) as sketch:
                    with Locations((0, -0.2 * self.key_h)):
                        Rectangle(6, 2)
                    fillet(sketch.vertices(), 0.999)
                extrude(amount=self.max_front_height - y - 2)
            key_body += bump_part.part

        # --- Legend Part Generation ---
        legend_parts: list[Part | None] = [None, None, None] # Center, TL, TR
        emboss_height = 1.0 # Set emboss height to 1mm

        # Find top face (needed for normals and points)
        top_face = None
        try:
            # Ensure we get the profile *without* the stem cutout for surface calculations
            outer_profile_no_shift = self._outer_key_profile() 
            top_face = outer_profile_no_shift.faces().sort_by(Axis.Z)[-1]
            top_center_on_face = top_face.center()
            # We still need a general Z reference, slightly above the absolute center
            base_z_ref = top_center_on_face.Z + self.eps 
        except (IndexError, RuntimeError):
            print(f"Warning: Could not reliably find top face for legends.")
            # Fallback Z, but normal/point finding will fail later if top_face is None
            base_z_ref = self.max_height

        # Helper function to create legend part
        def create_legend(text: str, font_size: float, position: tuple[float, float], alignment: tuple[Align, Align]) -> Part | None:
            if not text or top_face is None:
                return None
            
            # 1. Find target point on the actual surface
            #    Start with an approximate target point in 3D space
            approx_target_pt = Vector(position[0], position[1], base_z_ref)
            try:
                # Use closest_points with a temporary Vertex
                target_vertex = Vertex(approx_target_pt)
                # Attempt to get the closest point Vector directly
                closest_data = top_face.closest_points(target_vertex)
                
                # Check if the result is a Vector (or potentially a tuple containing one)
                if isinstance(closest_data, Vector):
                    surface_pt = closest_data
                elif isinstance(closest_data, tuple) and closest_data and isinstance(closest_data[0], Vector):
                     # Handle potential tuple return like (Vector(point_on_face), Vector(point_on_vertex))
                     surface_pt = closest_data[0]
                elif isinstance(closest_data, tuple) and closest_data and isinstance(closest_data[0], list) and closest_data[0] and isinstance(closest_data[0][0], Vector):
                    # Handle potential nested list return like ( [Vector(...)] , [...] )
                    surface_pt = closest_data[0][0]
                else:
                    raise RuntimeError(f"Unexpected result from closest_points: {type(closest_data)}")

                # Get the normal vector at that surface point
                surface_normal = top_face.normal_at(surface_pt)
            except (RuntimeError, IndexError, TypeError, AttributeError) as e: # Catch potential errors more broadly
                # Handle cases where finding points/normal might fail
                print(f"Warning: Could not find surface point/normal for legend '{text}': {e}. Skipping.")
                return None

            # 2. Create the tangent plane slightly offset outwards
            legend_origin = surface_pt + surface_normal * self.eps
            legend_plane = Plane(origin=legend_origin, z_dir=surface_normal)

            # 3. Build the text on the tangent plane and extrude
            with BuildPart() as part_builder:
                with BuildSketch(legend_plane):
                    Text(
                        text,
                        font_size=font_size,
                        font_style=FontStyle.BOLD,
                        align=alignment
                    )
                # Extrude along the plane's normal (which is the surface normal)
                extrude(amount=emboss_height)
            return part_builder.part

        # 1. Center Legend
        center_font_size = self.key_h * 0.4
        legend_parts[0] = create_legend(self.legend_center, center_font_size, (0, 0), (Align.CENTER, Align.CENTER))

        # 2. Top-Left Legend
        corner_font_size = self.key_h * 0.2
        tl_x = -self.key_w * 0.30
        tl_y = self.key_h * 0.30
        legend_parts[1] = create_legend(self.legend_top_left, corner_font_size, (tl_x, tl_y), (Align.CENTER, Align.CENTER))

        # 3. Top-Right Legend
        tr_x = self.key_w * 0.30
        tr_y = self.key_h * 0.30
        legend_parts[2] = create_legend(self.legend_top_right, corner_font_size, (tr_x, tr_y), (Align.CENTER, Align.CENTER))

        # --- Assign Colors (only if returning components) ---
        if return_components:
            key_body.color = Color(0.1, 0.1, 0.1) # Dark grey/Black (Body)
            legend_color_center = Color(0.9, 0.9, 0.9) # White (Center Legend)
            legend_color_corner = Color(0.7, 0.7, 0.9) # Light Blue (Corner Legends - example)

            if legend_parts[0]: legend_parts[0].color = legend_color_center
            if legend_parts[1]: legend_parts[1].color = legend_color_corner
            if legend_parts[2]: legend_parts[2].color = legend_color_corner


        # --- Return Logic ---
        if return_components:
            # Return separate body and list of legend parts
            return key_body, legend_parts
        else:
            # Return combined shape
            final_shape = key_body
            for part in legend_parts:
                if part:
                    final_shape += part
            return final_shape
