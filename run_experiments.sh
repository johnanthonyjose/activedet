if [[ $# -eq 0 ]] ; then
    echo 'Please provide the path of config file to run'
    exit 1
elif [[ $# -le 1 ]] ; then
    echo 'Please provide two arguments. Exiting...'
    exit 1
fi

configfile=$1
outdir=$2

# Trial 1
NOW=$(date +"%Y%m%d_%H%M%S")
output="$outdir/${NOW}"
python tools/active_lazy_train_net.py --num-gpu 1 --config-file "${configfile}" train.output_dir="${output}" train.seed=2863257
jq 'select( ."bbox/AP" != null) | ."bbox/AP"' "${output}/metrics.json"  > "${output}/AP.txt"
jq 'select( ."bbox/AP" != null) | ."bbox/AP50"' "${output}/metrics.json"  > "${output}/AP50.txt"
jq 'select( ."bbox/AP" != null) | ."bbox/AP75"' "${output}/metrics.json"  > "${output}/AP75.txt"

# Trial 2
NOW=$(date +"%Y%m%d_%H%M%S")
output="$outdir/${NOW}"
python tools/active_lazy_train_net.py --num-gpu 1 --config-file "${configfile}" train.output_dir="${output}" train.seed=55461164
jq 'select( ."bbox/AP" != null) | ."bbox/AP"' "${output}/metrics.json"  > "${output}/AP.txt"
jq 'select( ."bbox/AP" != null) | ."bbox/AP50"' "${output}/metrics.json"  > "${output}/AP50.txt"
jq 'select( ."bbox/AP" != null) | ."bbox/AP75"' "${output}/metrics.json"  > "${output}/AP75.txt"

# Trial 3
NOW=$(date +"%Y%m%d_%H%M%S")
output="$outdir/${NOW}"
python tools/active_lazy_train_net.py --num-gpu 1 --config-file "${configfile}" train.output_dir="${output}" train.seed=14509344
jq 'select( ."bbox/AP" != null) | ."bbox/AP"' "${output}/metrics.json"  > "${output}/AP.txt"
jq 'select( ."bbox/AP" != null) | ."bbox/AP50"' "${output}/metrics.json"  > "${output}/AP50.txt"
jq 'select( ."bbox/AP" != null) | ."bbox/AP75"' "${output}/metrics.json"  > "${output}/AP75.txt"

# Trial 4
NOW=$(date +"%Y%m%d_%H%M%S")
output="$outdir/${NOW}"
python tools/active_lazy_train_net.py --num-gpu 1 --config-file "${configfile}" train.output_dir="${output}" train.seed=3959168
jq 'select( ."bbox/AP" != null) | ."bbox/AP"' "${output}/metrics.json"  > "${output}/AP.txt"
jq 'select( ."bbox/AP" != null) | ."bbox/AP50"' "${output}/metrics.json"  > "${output}/AP50.txt"
jq 'select( ."bbox/AP" != null) | ."bbox/AP75"' "${output}/metrics.json"  > "${output}/AP75.txt"

# Trial 5
NOW=$(date +"%Y%m%d_%H%M%S")
output="$outdir/${NOW}"
python tools/active_lazy_train_net.py --num-gpu 1 --config-file "${configfile}" train.output_dir="${output}" train.seed=2120925
jq 'select( ."bbox/AP" != null) | ."bbox/AP"' "${output}/metrics.json"  > "${output}/AP.txt"
jq 'select( ."bbox/AP" != null) | ."bbox/AP50"' "${output}/metrics.json"  > "${output}/AP50.txt"
jq 'select( ."bbox/AP" != null) | ."bbox/AP75"' "${output}/metrics.json"  > "${output}/AP75.txt"

