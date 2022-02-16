outdir=$1
inputdir=$2
c=0
touch outfolder.txt
for pfolder in $(ls $inputdir)
do
	echo $pfolder
	echo "find $inputdir/$pfolder -type d -name "test-classes""
for testdir in $(find $inputdir/$pfolder -type d -name "test-classes")
do
	mkdir -p $outdir/$pfolder/$c
	echo " java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -i $testdir -g -o $outdir/$pfolder/$c "
        java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -i $testdir -g -o $outdir/$pfolder/$c
	echo "$outdir/$pfolder/$c,$testdir" >> outfolder.txt
	c=$((c+1))
done
done
