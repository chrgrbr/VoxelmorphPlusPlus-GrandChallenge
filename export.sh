#!/usr/bin/env bash

./build.sh

docker save voxelmorphplusplus | gzip -c > VoxelMorphPlusPlus.tar.gz
