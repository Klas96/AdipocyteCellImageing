# AdipocyteCellImageing

## Given
Gray scale 16-bit images with 65535 (2156 x 2556) levels of gray. The “color” of these images is determined by the spectral filters in the microscope so that while they are gray-scale images, they represent different parts of the color spectrum.

The data consists of images at three different magnification( 20, 40 and 60 times). Every magnification is contanied in separate data set.

The data consits of three classes: nuclei (blue), lipid droplets (green) and cell cytoplasm (red). (Backgorund maybe considered as a class?)

## Problem
With the brightfield images as input, predict areas of nuclei (blue), lipid droplets (green) and cell cytoplasm (red) in the fluorescense images. In other words, the fluorescense images are the labeling.

Every outputted images will be evaluated with three different metrics: morphological-, intensity- and texturalloss.
