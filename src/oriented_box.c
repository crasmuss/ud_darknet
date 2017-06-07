#include "oriented_box.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

oriented_box float_to_oriented_box(float *f)
{
    oriented_box b;
    b.x = f[0];
    b.y = f[1];
    b.w = f[2];
    b.theta = f[3];
    return b;
}

oriented_dbox oriented_derivative(oriented_box a, oriented_box b)
{
    oriented_dbox d;
    /*
    d.dx = 0;
    d.dw = 0;
    float l1 = a.x - a.w/2;
    float l2 = b.x - b.w/2;
    if (l1 > l2){
        d.dx -= 1;
        d.dw += .5;
    }
    float r1 = a.x + a.w/2;
    float r2 = b.x + b.w/2;
    if(r1 < r2){
        d.dx += 1;
        d.dw += .5;
    }
    if (l1 > r2) {
        d.dx = -1;
        d.dw = 0;
    }
    if (r1 < l2){
        d.dx = 1;
        d.dw = 0;
    }

    d.dy = 0;
    d.dh = 0;
    float t1 = a.y - a.h/2;
    float t2 = b.y - b.h/2;
    if (t1 > t2){
        d.dy -= 1;
        d.dh += .5;
    }
    float b1 = a.y + a.h/2;
    float b2 = b.y + b.h/2;
    if(b1 < b2){
        d.dy += 1;
        d.dh += .5;
    }
    if (t1 > b2) {
        d.dy = -1;
        d.dh = 0;
    }
    if (b1 < t2){
        d.dy = 1;
        d.dh = 0;
    }
    */
    return d;
}

float oriented_overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float oriented_box_intersection(oriented_box a, oriented_box b)
{
  /*
    float w = oriented_overlap(a.x, a.w, b.x, b.w);
    float h = oriented_overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
  */
  return 0.0;
}

float oriented_box_union(oriented_box a, oriented_box b)
{
  /*
    float i = oriented_box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
  */
  return 0.0;
}

float oriented_box_iou(oriented_box a, oriented_box b)
{
  return oriented_box_intersection(a, b)/oriented_box_union(a, b);
}

float oriented_box_rmse(oriented_box a, oriented_box b)
{
  /*
    return sqrt(pow(a.x-b.x, 2) + 
                pow(a.y-b.y, 2) + 
                pow(a.w-b.w, 2) + 
                pow(a.h-b.h, 2));
  */
    return sqrt(pow(a.x-b.x, 2) + 
                pow(a.y-b.y, 2) + 
                pow(a.w-b.w, 2) + 
                pow(a.theta-b.theta, 2));
}

oriented_dbox oriented_dintersect(oriented_box a, oriented_box b)
{
  /*
    float w = oriented_overlap(a.x, a.w, b.x, b.w);
    float h = oriented_overlap(a.y, a.h, b.y, b.h);
    oriented_dbox dover = oriented_derivative(a, b);
  */
    oriented_dbox di;

    /*
    di.dw = dover.dw*h;
    di.dx = dover.dx*h;
    di.dh = dover.dh*w;
    di.dy = dover.dy*w;
    //    di.dtheta = dover.dtheta*
    */
    
    return di;
}

oriented_dbox oriented_dunion(oriented_box a, oriented_box b)
{
    oriented_dbox du;

    /*
    oriented_dbox di = oriented_dintersect(a, b);
    du.dw = a.h - di.dw;
    du.dh = a.w - di.dh;
    du.dx = -di.dx;
    du.dy = -di.dy;
    */
    
    return du;
}


void test_oriented_dunion()
{
  /*
    oriented_box a = {0, 0, 1, 1};
    oriented_box dxa= {0+.0001, 0, 1, 1};
    oriented_box dya= {0, 0+.0001, 1, 1};
    oriented_box dwa= {0, 0, 1+.0001, 1};
    oriented_box dha= {0, 0, 1, 1+.0001};

    oriented_box b = {.5, .5, .2, .2};
    oriented_dbox di = oriented_dunion(a,b);
    printf("Union: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float inter =  oriented_box_union(a, b);
    float xinter = oriented_box_union(dxa, b);
    float yinter = oriented_box_union(dya, b);
    float winter = oriented_box_union(dwa, b);
    float hinter = oriented_box_union(dha, b);
    xinter = (xinter - inter)/(.0001);
    yinter = (yinter - inter)/(.0001);
    winter = (winter - inter)/(.0001);
    hinter = (hinter - inter)/(.0001);
    printf("Oriented Union Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
  */
}

void test_oriented_dintersect()
{
  /*
    oriented_box a = {0, 0, 1, 1};
    oriented_box dxa= {0+.0001, 0, 1, 1};
    oriented_box dya= {0, 0+.0001, 1, 1};
    oriented_box dwa= {0, 0, 1+.0001, 1};
    oriented_box dha= {0, 0, 1, 1+.0001};

    oriented_box b = {.5, .5, .2, .2};
    oriented_dbox di = oriented_dintersect(a,b);
    printf("Inter: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float inter =  oriented_box_intersection(a, b);
    float xinter = oriented_box_intersection(dxa, b);
    float yinter = oriented_box_intersection(dya, b);
    float winter = oriented_box_intersection(dwa, b);
    float hinter = oriented_box_intersection(dha, b);
    xinter = (xinter - inter)/(.0001);
    yinter = (yinter - inter)/(.0001);
    winter = (winter - inter)/(.0001);
    hinter = (hinter - inter)/(.0001);
    printf("Oriented Inter Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
  */
}

void test_oriented_box()
{
  /*
    test_oriented_dintersect();
    test_oriented_dunion();
    oriented_box a = {0, 0, 1, 1};
    oriented_box dxa= {0+.00001, 0, 1, 1};
    oriented_box dya= {0, 0+.00001, 1, 1};
    oriented_box dwa= {0, 0, 1+.00001, 1};
    oriented_box dha= {0, 0, 1, 1+.00001};

    oriented_box b = {.5, 0, .2, .2};

    float iou = oriented_box_iou(a,b);
    iou = (1-iou)*(1-iou);
    printf("%f\n", iou);
    oriented_dbox d = oriented_diou(a, b);
    printf("%f %f %f %f\n", d.dx, d.dy, d.dw, d.dh);

    float xiou = oriented_box_iou(dxa, b);
    float yiou = oriented_box_iou(dya, b);
    float wiou = oriented_box_iou(dwa, b);
    float hiou = oriented_box_iou(dha, b);
    xiou = ((1-xiou)*(1-xiou) - iou)/(.00001);
    yiou = ((1-yiou)*(1-yiou) - iou)/(.00001);
    wiou = ((1-wiou)*(1-wiou) - iou)/(.00001);
    hiou = ((1-hiou)*(1-hiou) - iou)/(.00001);
    printf("manual %f %f %f %f\n", xiou, yiou, wiou, hiou);
  */
}

oriented_dbox oriented_diou(oriented_box a, oriented_box b)
{
  /*
    float u = oriented_box_union(a,b);
    float i = oriented_box_intersection(a,b);
    oriented_dbox di = oriented_dintersect(a,b);
    oriented_dbox du = oriented_dunion(a,b);
    oriented_dbox dd = {0,0,0,0};

    if(i <= 0 || 1) {
        dd.dx = b.x - a.x;
        dd.dy = b.y - a.y;
        dd.dw = b.w - a.w;
        dd.dh = b.h - a.h;
        return dd;
    }

    dd.dx = 2*pow((1-(i/u)),1)*(di.dx*u - du.dx*i)/(u*u);
    dd.dy = 2*pow((1-(i/u)),1)*(di.dy*u - du.dy*i)/(u*u);
    dd.dw = 2*pow((1-(i/u)),1)*(di.dw*u - du.dw*i)/(u*u);
    dd.dh = 2*pow((1-(i/u)),1)*(di.dh*u - du.dh*i)/(u*u);
    return dd;
  */
}

typedef struct{
    int index;
    int class;
    float **probs;
} sortable_oriented_bbox;

int oriented_nms_comparator(const void *pa, const void *pb)
{
    sortable_oriented_bbox a = *(sortable_oriented_bbox *)pa;
    sortable_oriented_bbox b = *(sortable_oriented_bbox *)pb;
    float diff = a.probs[a.index][b.class] - b.probs[b.index][b.class];
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

void oriented_do_nms_obj(oriented_box *oriented_boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    sortable_oriented_bbox *s = calloc(total, sizeof(sortable_oriented_bbox));

    for(i = 0; i < total; ++i){
        s[i].index = i;       
        s[i].class = classes;
        s[i].probs = probs;
    }

    qsort(s, total, sizeof(sortable_oriented_bbox), oriented_nms_comparator);
    for(i = 0; i < total; ++i){
        if(probs[s[i].index][classes] == 0) continue;
        oriented_box a = oriented_boxes[s[i].index];
        for(j = i+1; j < total; ++j){
            oriented_box b = oriented_boxes[s[j].index];
            if (oriented_box_iou(a, b) > thresh){
                for(k = 0; k < classes+1; ++k){
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }
    free(s);
}


void oriented_do_nms_sort(oriented_box *oriented_boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    sortable_oriented_bbox *s = calloc(total, sizeof(sortable_oriented_bbox));

    for(i = 0; i < total; ++i){
        s[i].index = i;       
        s[i].class = 0;
        s[i].probs = probs;
    }

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            s[i].class = k;
        }
        qsort(s, total, sizeof(sortable_oriented_bbox), oriented_nms_comparator);
        for(i = 0; i < total; ++i){
            if(probs[s[i].index][k] == 0) continue;
            oriented_box a = oriented_boxes[s[i].index];
            for(j = i+1; j < total; ++j){
                oriented_box b = oriented_boxes[s[j].index];
                if (oriented_box_iou(a, b) > thresh){
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }
    free(s);
}

void oriented_do_nms(oriented_box *oriented_boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    for(i = 0; i < total; ++i){
        int any = 0;
        for(k = 0; k < classes; ++k) any = any || (probs[i][k] > 0);
        if(!any) {
            continue;
        }
        for(j = i+1; j < total; ++j){
            if (oriented_box_iou(oriented_boxes[i], oriented_boxes[j]) > thresh){
                for(k = 0; k < classes; ++k){
                    if (probs[i][k] < probs[j][k])
		      probs[i][k] = 0;
                    else
		      probs[j][k] = 0;
                }
            }
        }
    }
}

oriented_box encode_oriented_box(oriented_box b, oriented_box anchor)
{
    oriented_box encode;
    /*
    encode.x = (b.x - anchor.x) / anchor.w;
    encode.y = (b.y - anchor.y) / anchor.h;
    encode.w = log2(b.w / anchor.w);
    encode.h = log2(b.h / anchor.h);
    */
    return encode;
}

oriented_box decode_oriented_box(oriented_box b, oriented_box anchor)
{
    oriented_box decode;
    /*
    decode.x = b.x * anchor.w + anchor.x;
    decode.y = b.y * anchor.h + anchor.y;
    decode.w = pow(2., b.w) * anchor.w;
    decode.h = pow(2., b.h) * anchor.h;
    */
    return decode;
}
