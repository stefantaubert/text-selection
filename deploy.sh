#!/bin/bash

prog_name="text-selection"
cli_path=src/text_selection_app/cli.py

mkdir -p ./dist

pipenv run cxfreeze \
  -O \
  --compress \
  --target-dir=dist \
  --bin-includes="libffi.so" \
  --target-name=cli \
  $cli_path

echo "compiled."

if [ $1 ]
then
  cd dist
  zip $prog_name-linux.zip ./ -r
  cd ..
  echo "zipped."
fi
