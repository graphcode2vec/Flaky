outputdir=$(realpath $1)
input=$(realpath $2)
for d in $(ls $input)
do
[ -f $d ] && continue
echo $d
cd $input/$d
for f in $(find . -type d -name "test-classes")
do
  pf=$(dirname $f)
  echo $ouputdir/$d/$pf
  mkdir -p $outputdir/$d/$pf
  cp -rf $f $outputdir/$d/$pf
done
cd ..
done
