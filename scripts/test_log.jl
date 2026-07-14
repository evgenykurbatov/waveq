using Plots
x = -10:10
y = x.^2
p = plot(x, y, xscale=:log10)
savefig(p, "test_log.pdf")
