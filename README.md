#Running#
* Linux machine with CUDA
* Torch (http://torch.ch/docs/getting-started.html) with OpenCv bindings (https://github.com/VisionLabs/torch-opencv)
* Put the right path to the Locate dataset in testLib.lua at L5.
* Example command: `CUDA_VISIBLE_DEVICES=0 th testKeypoints.lua -s kanzelwandbahn`

#Notes#
* I hope you have some experience with Lua/Torch. Otherwise we can talk. If you run on Python, one can bridge the gap with https://github.com/hughperkins/pytorch .
* My code detects keypoints in images (opencv), extracts patches around them, computes matching score between them and evaluates correctness with average precision (correct = best match within 15px of the ground truth).
* Input images are loaded in `loadImagePair()` (one real (3 channels) and one synthetic (17 channels, various shadings)) and then network in `loadNetwork()`.
* `extractKeypoints()` crops patches around keypoints.
* `fullMatch()` computes pairwise matches between all synthetic and real keypoints (quadratic number of evaluations). The network returns a score for each pair of patches.
