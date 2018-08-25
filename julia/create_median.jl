using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "i"
        arg_type = Int
        required = true
        help = "Enter an integer i"
end
ps = parse_args(s)
i = ps["i"]

paths = ["from_zeros/gain_0.005", "from_zeros/gain_0.01", "from_RA/gain_0.005", "from_RA/gain_0.01", "from_HA/gain_0.005", "from_HA/gain_0.01"]
median_psi_vec = zeros(10_000, 3)
path = paths[i]
for t in 1:10_000
    println(t)
    median_psi_vec[t, :] = median(readdlm("../data/HA_learning/simulations/$path/psi/$t.csv", ','), 1)
    writedlm("../data/HA_learning/simulations/$path/median_psi/$t.csv", median_psi_vec[t, :], ',')
end
writedlm("../data/HA_learning/simulations/$path/median_psi/combined.csv", median_psi_vec, ',')
