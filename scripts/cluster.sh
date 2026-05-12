# Cluster sequences with MMseqs2

set -eu

high_poly_path="data/processed/fastas/high.fa"
low_poly_path="data/processed/fastas/low.fa"
output_dir="data/processed/clustered/"

cluster_params="\
-c 0.8 \
--cov-mode 0 \
--cluster-mode 0 \
"
cluster_seq_ids=(75 80 85 90 95)

at_exit() {
    if [ -e "$tmp_dir" ]; then
        rm -r "$tmp_dir"
    fi
}

tmp_dir="$output_dir/tmp/"

if [ -e "$output_dir" ]; then
    rm -r "$output_dir"
fi

mkdir -p "$output_dir"
mkdir -p "$tmp_dir"
trap at_exit EXIT

poly_db="$tmp_dir/poly_db"
cluster_db_prefix="$tmp_dir/cluster_db"

mmseqs createdb "$high_poly_path" "$low_poly_path" "$poly_db"
for seq_id in "${cluster_seq_ids[@]}"; do
    cluster_db="${cluster_db_prefix}_$seq_id"
    mmseqs cluster "$poly_db" "$cluster_db" "$tmp_dir" $cluster_params --min-seq-id 0.$seq_id
    mmseqs createtsv "$poly_db" "$poly_db" "$cluster_db" "$output_dir/clusters_$seq_id.tsv"
done
