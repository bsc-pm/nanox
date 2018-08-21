#Run this script from the base directory

docker run --rm -v "$PWD":/usr/src/nanox -w /usr/src/nanox bscpm/headache bash ./scripts/headerfy.sh
