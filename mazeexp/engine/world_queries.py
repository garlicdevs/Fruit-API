import math

import cocos.euclid as eu

class WorldQueries(object):
    """
    WorldQueries

    Methods for querying map inherited by WorldLayer
    Has context for game settings, map state and player state
    """

    def __init__(self):
        super(WorldQueries, self).__init__()

    def distance_to_tile(self, point, direction, length = 50):
        """
        Find nearest wall on a given bearing.
        Used for agent wall sensors.
        """
        assert isinstance(point, eu.Vector2)
        assert isinstance(direction, int) or isinstance(direction, float)
        assert isinstance(length, int) or isinstance(length, float)

        # Recursive dead-reckoning to next tile
        # Given `point`, look for where intersects with next boundary (`y % 10`) in `direction`
        def search_grid(search, rad, distance = 0, depth = 10):
            assert isinstance(search, eu.Vector2)
            assert isinstance(rad, float)

            if depth == 0:
                return distance
            depth -= 1

            # Exit if outside window.
            if abs(search.x) > self.width or abs(search.y) > self.height:
                return distance

            m = math.tan(rad) # Slope
            sin = math.sin(rad)
            cos = math.cos(rad)
            #print(sin, cos)

            top    = (cos > 0)
            bottom = (cos < 0)
            left   = (sin < 0)
            right  = (sin > 0)

            start  = eu.Vector2(search.x, search.y)
            ends   = eu.Vector2()

            # Helper function
            # FIXME: Does MapLayer provide something better? Neighbours?
            # Find next grid on given axis
            def get_boundary(axis, increasing):
                assert (isinstance(axis, str) or isinstance(axis, unicode)) and (axis == 'x' or axis == 'y')

                if axis == 'x':
                    tile = self.map_layer.tw
                    position = search.x
                elif axis == 'y':
                    tile = self.map_layer.th
                    position = search.y

                # Set bound to next tile on axis
                # Offset next search by one pixel into tile
                bound = (position % tile)
                if increasing:
                    bound = tile - bound
                    bound = position + bound
                    offset = 1
                else:
                    bound = position - bound
                    offset = -1

                # Find intersect
                if axis == 'x':
                    intersect = ((bound - search.x) / m) + search.y
                    return eu.Vector2(bound+offset, intersect)
                elif axis == 'y':
                    intersect = -m * (search.y - bound) + search.x
                    return eu.Vector2(intersect, bound+offset)
            # End Helper

            if top or bottom:
                ends.y = get_boundary('y', top)
                ends.y.y = min(ends.y.y, self.height)

            if left or right:
                ends.x = get_boundary('x', right)
                ends.x.x = min(ends.x.x, self.width)

            # Get shortest collision between axis
            lengths = eu.Vector2(0, 0)
            if type(ends.x) == eu.Vector2:
                diff = start - ends.x
                lengths.x = math.sqrt(diff.dot(diff))
            if type(ends.y) == eu.Vector2:
                diff = start - ends.y
                lengths.y = math.sqrt(diff.dot(diff))

            end = None

            # Find shortest boundary intersect
            index_min = min(range(len(lengths)), key=lengths.__getitem__)

            if lengths[index_min] > 0:
                distance += lengths[index_min]
                end = ends[index_min]

            if end:
                cell = self.map_layer.get_at_pixel(end.x, end.y)
                if not cell or not cell.tile or not cell.tile.id > 0:
                    # Recurse
                    return search_grid(end, rad, distance, depth)

            return distance
        # End Helper

        # Start at `point`, check tile under each pixel
        return search_grid(point, direction)
