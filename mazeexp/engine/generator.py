from random import randint
import cocos.tiles as ti

import os
script_dir = os.path.dirname(__file__)

HORIZONTAL = 0
VERTICAL = 1

class Generator():
    """
    Generator

    Maze map generation
    """

    def map(self, width, height):
        """
        Creates and returns a new randomly generated map
        """
        template = ti.load(os.path.join(script_dir, 'assets', 'template.tmx'))['map0']
        #template.set_view(0, 0, template.px_width, template.px_height)
        template.set_view(0, 0, width*template.tw, height*template.th)

        # TODO: Save the generated map.
        #epoch = int(time.time())
        #filename = 'map_' + str(epoch) + '.tmx'

        # Draw borders
        border_x = template.cells[width]
        for y in range(0,height+1):
            border_x[y].tile = template.cells[0][0].tile

        for x in range(0,width):
            template.cells[x][height].tile = template.cells[0][0].tile

        # Start within borders
        #self.recursive_division(template.cells, 3, (template.px_width/template.tw)-1, (template.px_height/template.th)-1, 0, 0)
        self.recursive_division(template.cells, 3, width, height, 0, 0)

        return template

    def recursive_division(self, cells, min_size, width, height, x=0, y=0, depth=0):
        """
        Recursive division:
            1. Split room randomly
                1a. Dodge towards larger half if in doorway
            2. Place doorway randomly
            3. Repeat for each half
        """
        assert isinstance(cells, list)
        assert isinstance(min_size, int) or isinstance(min_size, float)
        assert isinstance(width, int) or isinstance(width, float)
        assert isinstance(height, int) or isinstance(height, float)
        assert isinstance(x, int) or isinstance(x, float)
        assert isinstance(y, int) or isinstance(y, float)
        assert isinstance(depth, int)

        if width <= min_size or height <= min_size:
            return

        # Choose axis to divide
        if width < height:
            axis = VERTICAL
        elif height < width:
            axis = HORIZONTAL
        else:
            axis = randint(0,1)

        cut_size = height
        gap_size = width
        if axis == HORIZONTAL:
            cut_size = width
            gap_size = height

        if cut_size-min_size < min_size:
            #print('min cut')
            return
        if gap_size-min_size < min_size:
            #print('min gap')
            return

        # Random division and doorway
        cut = randint(min_size, cut_size-min_size)
        gap = randint(min_size, gap_size-min_size)

        if not (cut > 0 and gap > 0):
            #print('Reached zero sized cell')
            return

        # Check if next tile is a doorway
        def dodge_doors(cut):
            assert isinstance(cut, int) or isinstance(cut, float)

            empty = False
            if axis == HORIZONTAL:
                idx = x+gap_size
                #print(idx,y+cut)
                door = cells[idx][y+cut]
                empty = empty or not door or not door.tile
                #door.tile = cells[49][1].tile

                idx = x
                #print(idx,y+cut)
                door = cells[idx][y+cut]
                empty = empty or not door or not door.tile
                #door.tile = cells[49][0].tile
            else:
                idx = y+gap_size
                #print(x+cut, idx)
                door = cells[x+cut][idx]
                empty = empty or not door or not door.tile
                #door.tile = cells[49][0].tile
                idx = y
                #print(x+cut,idx)
                door = cells[x+cut][idx]
                empty = empty or not door or not door.tile
                #door.tile = cells[49][1].tile

            # Try again on longest side
            if empty:
                #print('Door', idx, cut)
                if gap + (min_size / 2) > (gap_size / 2) - (min_size / 2):
                    cut -= 1
                else:
                    cut += 1

                if cut < min_size or cut > cut_size-min_size:
                    #print('Reached minimum size')
                    return None
                else:
                    return dodge_doors(cut)

            return cut

        # Skip doors check first time around
        if depth > 0:
            cut = dodge_doors(cut)
            if cut is None:
                #print('No viable cut found')
                return None
        depth += 1

        # Create new wall tiles
        for i in range(0, gap_size):
            if abs(gap - i) > 0:
                # Copy wall tile from (0,0)
                if axis == HORIZONTAL:
                    cells[x+i][y+cut].tile = cells[0][0].tile
                else:
                    cells[x+cut][y+i].tile = cells[0][0].tile

        # Recurse into each half
        #print(x, y, [cut, gap], [cut_size, gap_size], 'H' if (axis == HORIZONTAL) else 'V')

        # N
        nx, ny = x, y
        w, h = [cut, height] if (axis == HORIZONTAL) else [width, cut]
        self.recursive_division(cells, min_size, w, h, nx, ny, depth)

        # S
        nx, ny = [x+cut, y] if (axis != HORIZONTAL) else [x, y+cut]
        w, h = [cut_size-cut, height] if (axis == HORIZONTAL) else [width, cut_size-cut]
        self.recursive_division(cells, min_size, w, h, nx, ny, depth)
