== WHAT IS IT? ==
It's a CUDA implementation of Canny algorithm(http://en.wikipedia.org/wiki/Canny_edge_detector).

== BASED ON ==
* Y. Luo, R. Duraiswami. ’Canny Edge Detection on NVIDIA CUDA’. CVPR, 2008
* O. Stava, B. Benes. ’Connected Component Labeling in CUDA’. GPU Computing Gems Emerald Edition. 2011

== USAGE ==
./canny <file.pgm> <threshold1> <treshold2>

== LIMITATIONS ==
Input file must be a grayscale image in PGM format. It's height and width must be divisible by 16.
