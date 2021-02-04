docker run -u $(id -u):$(id -g) --rm --gpus all -it \
 -v /home3/pol/data/:/data:rw \
 -v /home3/pol/dataset_scannet:/database:rw \
 -v $(pwd):/project:rw \
 -w /project \
  deepmvsforked