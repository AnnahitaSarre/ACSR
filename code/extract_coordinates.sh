

for gender in 'male' 'female'; do
	for cropping in 'cropped' 'non_cropped'; do
		cmd='python extract_coordinates.py --gender '$gender' --cropping '$cropping
		echo $cmd
		eval $cmd
	done
done
