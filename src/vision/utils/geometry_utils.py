from shapely.geometry import Polygon

def load_zone_polygon(camera_id: int, mode: str) -> Polygon:
    # A mocked polygon, could be loaded from config a JSON file.
    return Polygon([(50, 50), (600, 50), (600, 400), (50, 400)])
