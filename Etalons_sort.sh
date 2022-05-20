#!/bin/sh
cd "$(dirname "$0")"

mkdir Pictures_sorted
out="$PWD/Pictures_sorted"

cd "Pictures-mast-etalons"
dir -1N . | while read d; do
    mkdir "$out/${d}_KOTEVNI" "$out/${d}_NOSNY"
    dir -1N "$d" | while read f; do
        subtp="$(echo "$f" | awk 'BEGIN{FS="_"; ORS=""}{print $2}')"
#         if [ "$subtp" == "KOTEVNI" ]; then
#             outfld="$out/${f}_KOTEVNI"
#         else
#             outfld="$out/${f}_NOSNY"
#         fi
        crout="$out/${d}_${subtp}"
        echo "$crout"
        mkdir "$crout"
        cp "$d/$f" "$crout"
    done
done
