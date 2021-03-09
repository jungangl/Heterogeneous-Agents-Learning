using Plots, StatsBase
iid = "iid"
period = 10000(iid == "iid") + 4008(iid != "iid")
#t_vec = sample(period - 2000:period, 100, replace = false)
str = "../data/HA/yearly/$(iid)_high_amin/learning/seed1/simulations/from_rational_RA/gain_0.10"
psi_vec = readdlm("$str/mean_psi/combined.csv", ',')
psi_RE = readdlm("../data/HA/yearly/$(iid)_high_amin/rational/psi.csv", ',')
a = readdlm("$str/a/$period.csv", ',')
s = readdlm("$str/s/$period.csv", ',')
psi = readdlm("$str/psi/$period.csv", ',')



states = if iid == "iid"
    [1 5 10 15 21]
elseif iid == "noiid"
    [1 5 11]
end


p1 = scatter(title = "", layout = (length(states), 1), size = (1400, length(states) * 300))
p2 = scatter(title = "", layout = (length(states), 1), size = (1400, length(states) * 300))


for (i, state) in enumerate(states)
    indx = find(s .== state)
    scatter!(p1, a[indx], psi[indx, 2], xlabel = "s = $state", label = "", subplot = i)
    plot!(p1, x -> mean(psi[:, 2]), 0., maximum(a[indx]), label = "", lw = 2, subplot = i)
    scatter!(p2, a[indx], psi[indx, 3], xlabel = "s = $state", label = "", subplot = i)
    plot!(p2, x -> mean(psi[:, 3]), 0., maximum(a[indx]), label = "", lw = 2, subplot = i)
end

p3 = plot(grid = false, layout = (3, 1))
for i in 1:3
    p3 = plot!(psi_vec[:, i], label = "", title = "\\psi$(i)", subplot = i)
    plot!(p3, ones(size(psi_vec, 1)) .* psi_RE[i], label = "", subplot = i)
end



savefig(p1, "$(iid)_loading_on_asset.pdf")
savefig(p2, "$(iid)_loading_on_aggregate.pdf")
savefig(p3, "$(iid)_psi.pdf")
