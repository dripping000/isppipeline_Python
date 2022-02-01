from Grid import *


class Patch(Grid):

    # Build a patch object
    def __init__(self, center, half_size, image):
        Grid.__init__(self, half_size * 2 + 1, half_size * 2 + 1)
        self.center = center
        self.half_size = half_size
        self.fill_grid(self.center, image)
    # end def

    # Set the value in the grid
    def fill_grid(self, center, image):
        # Patch width
        for x in range(self.abscissa_size):
            # Patch height
            for y in range(self.ordinate_size):
                if (
                    center[0] + x - self.half_size >= 0                     # Take values only in the image
                    and center[1] + y - self.half_size >= 0                 # Warning :
                    and center[0] + x - self.half_size < image.width        #   - out of range without
                    and center[1] + y - self.half_size < image.height       #     this checkup
                ):
                    self.grid[x][y] = image.getpixel((center[0] + x - self.half_size, center[1] + y - self.half_size))
        #print(self.grid)
                 # end if
        # end for
    # end def

    # Compare a grid with the current grid
    def compare_grid(self, grid):
        distance = 0
        # Patch width
        for x in range(self.abscissa_size):
            # Patch height
            for y in range(self.ordinate_size):
                if (
                    type(grid) is Patch             # Check types before comparison
                    and grid.grid[x][y] != -1       # Check the content of the other grid
                    and self.grid[x][y] != -1       # Check the content of the current grid
                ):
                    # Compute the distance
                    distance += abs(grid.grid[x][y] - self.grid[x][y]) ** 2
                # end if
            # end for
        # end for
        return distance
    # end def

# end class
