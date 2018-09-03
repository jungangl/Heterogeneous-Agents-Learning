using ArgParse
## Combine all of the csv files


function combine_csv(path, file, with)
    dir = "../data/HA/$with/learning/simulations/$path/$file"
    width = size(readdlm("$dir/1.csv", ','), 2)
    var = zeros(10_000, width)
    for t in 1:10_000
        var[t, :] = readdlm("$dir/$t.csv", ',')
    end
    writedlm("$dir/combined.csv", var, ',')
end


function dim1to2(i, N)
    comp1 = 0
    comp2 = 0
    if mod(i, N) == 0
        comp1 = i ÷ N
        comp2 = N
    else
        comp1 = i ÷ N + 1
        comp2 = mod(i, N)
    end
    return comp1, comp2
end


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
files = ["mean_s", "mean_psi", "median_psi", "mean_nu_bar_c", "mean_nu_bar", "mean_nu", "mean_n", "mean_c", "mean_a", "w", "theta", "r"]
total_i = length(paths) * length(files)
N = length(files)
comp1, comp2 = dim1to2(i, N)
combine_csv(paths[comp1], files[comp2], "with_iid")
