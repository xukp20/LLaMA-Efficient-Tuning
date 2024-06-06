# projection_type, loop_layers, loop_times, fix_projection, projection_init, fix_llm
# bash run_meta_local_loop_phi.sh vec 8-24 1 0 avg 0 42

# v2: update_style, loop_layers, loop_times, seed
bash run_meta_local_loop_phi_v2.sh loop 8-24 2 42
bash run_meta_local_loop_phi_v2.sh res 8-24 2 42