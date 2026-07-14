using FourierTools

# A 1D example
x = range(-5, 5, length=11)
dx = step(x)

# I want frequencies xi from -2 to 2 with length 15.
# Let's say dxi = 4 / 14
# the formula for czt frequency points is something like: k * dxi.
# By default, what does scale=1.0 mean?
# Let's see what FourierTools czt scale actually does.
