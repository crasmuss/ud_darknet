#ifndef ORIENTED_BOX_H
#define ORIENTED_BOX_H

#define ORIENTED_BOX_HEIGHT 
typedef struct{
  float x, y, w, theta;
} oriented_box;

typedef struct{
  float dx, dy, dw, dtheta;
} oriented_dbox;

oriented_box float_to_oriented_box(float *f);
float oriented_box_iou(oriented_box a, oriented_box b);
float oriented_box_rmse(oriented_box a, oriented_box b);
oriented_dbox oriented_diou(oriented_box a, oriented_box b);
void oriented_do_nms(oriented_box *oriented_boxes, float **probs, int total, int classes, float thresh);
void oriented_do_nms_sort(oriented_box *oriented_boxes, float **probs, int total, int classes, float thresh);
void oriented_do_nms_obj(oriented_box *oriented_boxes, float **probs, int total, int classes, float thresh);
oriented_box oriented_decode_box(oriented_box b, oriented_box anchor);
oriented_box oriented_encode_box(oriented_box b, oriented_box anchor);

#endif
