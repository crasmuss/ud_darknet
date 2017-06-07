#ifndef ORIENTED_REGION_LAYER_H
#define ORIENTED_REGION_LAYER_H

#include "layer.h"
#include "network.h"

layer make_oriented_region_layer(int batch, int h, int w, int n, int classes, int coords);
void forward_oriented_region_layer(const layer l, network_state state);
void backward_oriented_region_layer(const layer l, network_state state);
void get_oriented_region_boxes(layer l, int w, int h, float thresh, float **probs, oriented_box *boxes, int only_objectness, int *map, float tree_thresh);
void resize_region_layer(layer *l, int w, int h);

#ifdef GPU
void forward_oriented_region_layer_gpu(const layer l, network_state state);
void backward_oriented_region_layer_gpu(layer l, network_state state);
#endif

#endif
