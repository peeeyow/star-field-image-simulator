import importlib.resources


with importlib.resources.path(
    "star_field_image_simulator.data", "star_catalog.db"
) as database_path:
    DATABASE_PATH = str(database_path)
