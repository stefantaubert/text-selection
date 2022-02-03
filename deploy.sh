#!/bin/bash

prog_name="text-selection"
cli_path=src/text_selection_app/cli.py

mkdir ./dist
deactivate
pipenv run cxfreeze \
  -O \
  --compress \
  --target-dir=dist \
  --bin-includes="libffi.so" \
  --target-name=$prog_name \
  $cli_path

echo "compiled."
# copy to local apps folder
mkdir -p /home/mi/apps/$prog_name
cp dist/* -r /home/mi/apps/$prog_name
echo "deployed."

if [ $1 ]
then
  cd dist
  zip $prog_name-linux.zip ./ -r
  cd ..
  echo "zipped."
fi
