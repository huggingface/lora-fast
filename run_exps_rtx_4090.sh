

python run_benchmark.py --disable_fa3 --quantize_t5 --out_dir=no_fa3  2>& 1 | tee no_fa3.txt

python run_benchmark.py --disable_fa3 --quantize_t5 \
    --disable_compile --out_dir=no_compile_fa3  2>& 1 | tee no_compile_fa3.txt

python run_benchmark.py --disable_fa3 --disable_fp8 --offload \
    --disable_recompile_error --out_dir=no_fa3_fp8_nf4 2>& 1 | tee no_fa3_fp8_nf4.txt

python run_benchmark.py --disable_fa3 --disable_fp8 --quantize_t5 --offload \
    --disable_recompile_error --out_dir=no_fa3_fp8 2>& 1 | tee no_fa3_fp8.txt

python run_benchmark.py --disable_fa3 --disable_fp8 --offload \
    --disable_recompile_error --disable_compile \
    --out_dir=no_fa3_fp8_nf4_compile 2>& 1 | tee no_fa3_fp8_nf4_compile.txt

python run_benchmark.py --disable_fa3 --disable_fp8 --quantize_t5 --offload \
    --disable_recompile_error --disable_compile \
    --out_dir=no_fa3_fp8_compile 2>& 1 | tee no_fa3_fp8_compile.txt

python run_benchmark.py \
    --disable_fa3 --disable_fp8 --offload \
    --disable_hotswap \
    --disable_recompile_error \
    --out_dir=no_fa3_fp8_nf4_hotswap 2>& 1 | tee no_fa3_fp8_nf4_hotswap.txt

python run_benchmark.py \
    --disable_fa3 --disable_fp8 --offload --quantize_t5 \
    --disable_hotswap \
    --disable_recompile_error \
    --out_dir=no_fa3_fp8_hotswap 2>& 1 | tee no_fa3_fp8_hotswap.txt

python run_benchmark.py \
    --disable_fa3 --disable_fp8 --offload \
    --disable_hotswap \
    --disable_compile \
    --disable_recompile_error \
    --out_dir=no_fa3_fp8_nf4_hotswap_comp 2>& 1 | tee no_fa3_fp8_nf4_hotswap_comp.txt

python run_benchmark.py \
    --disable_fa3 --disable_fp8 --offload --quantize_t5 \
    --disable_hotswap \
    --disable_compile \
    --disable_recompile_error \
    --out_dir=no_fa3_fp8_hotswap_comp 2>& 1 | tee no_fa3_fp8_hotswap_comp.txt

