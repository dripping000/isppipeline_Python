class Grid:

    # Build a grid object
    def __init__(self, abscissa_size, ordinate_size):
        self.abscissa_size = abscissa_size
        self.ordinate_size = ordinate_size
        self.grid = self.init_grid()
    # end def

    # Set the value to -1 by default for initialize the array with the good size
    def init_grid(self):
        grid = [[-1 for y in range(self.abscissa_size)] for x in range(self.ordinate_size)]
        return grid
    # end def

# end class
