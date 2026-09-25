oex export
==========

Generated:        2026-05-12 12:16:38 UTC
oex version:      0.2.1
Project:          https://github.com/osgeonepal/oex

Country (ISO3):   IND
Boundary:         geoBoundaries CGAZ ADM0 (buffered +5000m)
Bounding box:     (68.1336, 6.7707, 97.4388, 37.0295)

Dataset:          railways
Format:           ESRI Shapefile (shp)
Features:         107,081

Source:           OpenStreetMap contributors
Source URL:       https://www.openstreetmap.org/
Snapshot:         2026-05-10
License:          hdx-odc-odbl
License URL:      https://opendatacommons.org/licenses/odbl/1-0/

About the source
  OpenStreetMap is a community-edited geographic dataset of the world. Country
  features are extracted from the source PBF via quackosm with the union of
  all category tag filters; per-category exports apply tag predicates at query
  time.

Notes
  - Shapefile output is split by geometry type:
    <category>_polygons.shp, <category>_lines.shp, <category>_points.shp.
    This is a shapefile-format limitation, not a data limitation.
  - Field names are truncated to 10 characters in shp; gpkg keeps them full.

Feedback:         https://github.com/osgeonepal/oex/issues
Engine: geofabrik