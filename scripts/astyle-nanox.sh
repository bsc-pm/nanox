OPTIONS="-s3vfwNlSKOCLD"
file=$1

for file in $(find . -name *.hpp); do
	astyle $OPTIONS $file
done

for file in $(find . -name *.cpp); do
	astyle $OPTIONS $file
done

for file in $(find . -name *.h); do
	astyle $OPTIONS $file
done

for file in $(find . -name *.c); do
	astyle $OPTIONS $file
done

