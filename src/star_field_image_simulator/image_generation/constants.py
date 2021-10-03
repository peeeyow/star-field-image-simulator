import importlib.resources


with importlib.resources.path(
    "star_field_image_simulator.data", "star_catalog.db"
) as database_path:
    DATABASE_PATH = str(database_path)

TABLE_NAME = "star_catalog"
ALPHA_MIN = 0
ALPHA_MAX = 360
DELTA_MIN = -90
DELTA_MAX = 90
HALF_REVOLUTION = 180
REL = 1e-4
U_COORDINATE_ORIGIN = 0
V_COORDINATE_ORIGIN = 0
