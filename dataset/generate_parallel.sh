set -e

for (( c=1; c<=$1; c++ ))
do
    python3 generate.py -c $2 -n $3 -o $4.$c --cf /tmp/watusi$c > data/p$c.log &
done 

echo "Waiting for dataset generation to finish. Enter to resume."
read var

cat data/$4.*.hl > data/$4.hl
cat data/$4.*.ll > data/$4.ll
cat data/$4.*.meta > data/$4.meta
cat data/p*.log > data/$4.log

rm data/$4.*.hl
rm data/$4.*.ll
rm data/$4.*.meta
rm data/p*.log