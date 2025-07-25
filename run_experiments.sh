python run_benchmark.py --out_dir=results/optimized  2>& 1 | tee results/optimized.txt

python run_benchmark.py --disable_compile --out_dir=results/no_compile  2>& 1 | tee results/no_compile.txt

python run_benchmark.py --disable_compile --disable_fp8 --out_dir=results/no_compile_fp8  2>& 1 | tee results/no_compile_fp8.txt

python run_benchmark.py --disable_fa3 --out_dir=results/no_fa3 2>& 1 | tee results/no_fa3.txt

python run_benchmark.py --disable_fa3 --disable_fp8 --out_dir=results/no_fa3_fp8 2>& 1 | tee results/no_fa3_fp8.txt

# Errors
# python run_benchmark.py \
#     --disable_fa3 --disable_fp8 \
#     --disable_hotswap \
#     --out_dir=results/no_fa3_fp8_hotswap 2>& 1 | tee results/no_fa3_fp8_hotswap.txt

python run_benchmark.py \
    --disable_fa3 --disable_fp8 \
    --disable_hotswap \
    --disable_recompile_error \
    --out_dir=results/no_fa3_fp8_hotswap_no_trigger 2>& 1 | tee results/no_fa3_fp8_hotswap_no_trigger.txt

python run_benchmark.py \
    --disable_fa3 --disable_fp8 \
    --disable_hotswap \
    --disable_compile \
    --out_dir=results/no_fa3_fp8_hotswap_comp 2>& 1 | tee results/no_fa3_fp8_hotswap_comp.txt
