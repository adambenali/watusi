folder="."

if [ ! -z "$1" ]; then
    folder="$1"
fi;

cd $folder

echo "Working inside $folder"
echo "Building vocabularies .."
onmt_build_vocab -c config.yaml

echo "Training .."
onmt_train -config config.yaml