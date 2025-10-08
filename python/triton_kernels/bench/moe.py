from bench_mlp import roofline_mlp, bench_mlp

dim1 = 5760
dim2 = 5760
n_expts_tot = 128
n_expts_act = 4
batch_sizes = [32]

# 1. fp8x-fp8w-TP1-EP1
roofline_mlp(batch_sizes, dim1, dim2, n_expts_tot, n_expts_act, 
             "fp8", "fp8", TP=1, EP=1, name="gpt-oss-x2")

# 2. fp8x-mx4w-TP1-EP1
roofline_mlp(batch_sizes, dim1, dim2, n_expts_tot, n_expts_act, 
             "fp8", "mx4", TP=1, EP=1, name="gpt-oss-x2")
