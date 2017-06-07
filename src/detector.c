#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif
static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};

void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network *nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = parse_network_cfg(cfgfile);
	printf("num layers = %i (gpu %i)\n", nets[i].n, i);
        if(weightfile){
	  if (clear)
	    load_weights_upto(&nets[i], weightfile, nets[i].n - 1);
	  else
	    load_weights(&nets[i], weightfile);
	  //printf("current batch %i, max batch %i\n", get_current_batch(nets[0]), nets[0].max_batches);
        }
        if(clear) *nets[i].seen = 0;
        nets[i].learning_rate *= ngpus;
    }
    srand(time(0));
    network net = nets[0];

    int imgs = net.batch * net.subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    data train, buffer;

    layer l = net.layers[net.n - 1];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    args.threads = 8;

    args.angle = net.angle;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;

    pthread_t load_thread = load_data(args);
    clock_t time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int dim = (rand() % 10 + 10) * 32;
            if (get_current_batch(net)+200 > net.max_batches) dim = 608;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            for(i = 0; i < ngpus; ++i){
                resize_network(nets + i, dim, dim);
            }
            net = nets[0];
        }
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        /*
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           if(!b.x) break;
           printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
           }
           image im = float_to_image(448, 448, 3, train.X.vals[10]);
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           printf("%d %d %d %d\n", truth.x, truth.y, truth.w, truth.h);
           draw_bbox(im, b, 8, 1,0,0);
           }
           save_image(im, "truth11");
         */

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}


static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '_');
    return atoi(p+1);
}

static void print_cocos(FILE *fp, char *image_path, box *boxes, float **probs, int num_boxes, int classes, int w, int h)
{
    int i, j;
    int image_id = get_coco_image_id(image_path);
    for(i = 0; i < num_boxes; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, probs[i][j]);
        }
    }
}

void print_detector_detections(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void print_imagenet_detections(FILE *fp, int id, box *boxes, float **probs, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            int class = j;
            if (probs[i][class]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j+1, probs[i][class],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    printf("a\n");
    
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            get_region_boxes(l, w, h, thresh, probs, boxes, 0, map, .5);
            if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, classes, nms);
            if (coco){
                print_cocos(fp, path, boxes, probs, l.w*l.h*l.n, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, boxes, probs, l.w*l.h*l.n, classes, w, h);
            } else {
                print_detector_detections(fps, id, boxes, probs, l.w*l.h*l.n, classes, w, h);
            }
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

// note that test set is hard-coded

void validate_detector_recall(char *cfgfile, char *weightfile, float thresh)
{
  printf("validate_detector_recall()\n");
  
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    //    list *plist = get_paths("data/voc.2007.test");
    list *plist = get_paths("data/scallop_test.txt");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;

    int j, k;
    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

    /*    float thresh = .001; */
    float iou_thresh = .5;
    float nms = .4;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;
    int wrong = 0;

    // loop over test images
    
    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        get_region_boxes(l, 1, 1, thresh, probs, boxes, 1, 0, .5);
        if (nms) do_nms(boxes, probs, l.w*l.h*l.n, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < l.w*l.h*l.n; ++k){
            if(probs[k][0] > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < l.w*l.h*l.n; ++k){
                float iou = box_iou(boxes[k], t);
                if(probs[k][0] > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%4d %4d %4d %4d  RPs/Img: %.2f\tIOU: %.2f%% R %.2f%% P %.2f%% [thresh = %.05f]\n", i, correct, total, proposals, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total, 100.*(float)correct/(float)proposals, thresh);
        free(id);
        free_image(orig);
        free_image(sized);
    }  // done with one image, on to next
    
}

void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.4;

    list *plist = get_paths("data/scallop_test.txt");
    char **paths = (char **)list_to_array(plist);

    printf("testing detector with thresh %f\n", thresh);
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        layer l = net.layers[net.n-1];

        box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));

        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh);
        if (l.softmax_tree && nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
	draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);

        save_image(im, "predictions");
        show_image(im, "predictions");

        free_image(im);
        free_image(sized);
        free(boxes);
        free_ptrs((void **)probs, l.w*l.h*l.n);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}

int haveWindow = 0;

int image_number = 0;
char *image_name;

#ifdef OPENCV
void my_show_image_cv(image p, const char *name, int do_save)
{
    int x,y,k;
    image copy = copy_image(p);
    constrain_image(copy);
    if(p.c == 3) rgbgr_image(copy);
    //normalize_image(copy);

    char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
    sprintf(buff, "%s", name);

    IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
    int step = disp->widthStep;
    if (!haveWindow) {
      //      cvNamedWindow(buff, CV_WINDOW_NORMAL);
      cvNamedWindow(buff, CV_WINDOW_AUTOSIZE);
      haveWindow = 1;
    }
    //cvMoveWindow(buff, 100*(windows%10) + 200*(windows/10), 100*(windows%10));
    for(y = 0; y < p.h; ++y){
        for(x = 0; x < p.w; ++x){
            for(k= 0; k < p.c; ++k){
                disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(copy,x,y,k)*255);
            }
        }
    }
    free_image(copy);

    cvShowImage(buff, disp);

    //    xxxcvSaveImage(disp);

    if (do_save) {
      image_name = (char *) malloc(256*sizeof(char));
      sprintf(image_name, "/tmp/output_%06i.png", image_number++);
      printf("%s\n", image_name);
    
      cvSaveImage(image_name, disp, 0);
    }
    
    cvReleaseImage(&disp);
}
#endif

// test images are hard-coded

void my_test_detector(char *datacfg, char *cfgfile, char *weightfile, float thresh, int do_draw, float *precision, float *recall, int *num_proposals, int *num_correct, int *num_total)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();

    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    if (do_draw)
      fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    //    list *plist = get_paths("data/voc.2007.test");
    list *plist = get_paths("data/scallop_test.txt");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;

    int j, k;
    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

    /*    float thresh = .001; */
    float iou_thresh = .5;
    float nms = .5 * iou_thresh;   // .4

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;
    int wrong = 0;

    if (do_draw)
      printf("MY testing detector with thresh %f\n", thresh);

    char *outfilename = (char *) malloc(256 * sizeof(char));
    
    for (int i = 0; i < m; i++ ) {

      char *path = paths[i];

      if (do_draw)
	printf("%s\n", path);
      
      image orig = load_image_color(path, 0, 0);
      image sized = resize_image(orig, net.w, net.h);
      char *id = basecfg(path);
      network_predict(net, sized.data);
      get_region_boxes(l, 1, 1, thresh, probs, boxes, 1, 0, .5);
      if (nms)
	do_nms(boxes, probs, l.w*l.h*l.n, 1, nms);
      
      char labelpath[4096];
      find_replace(path, "images", "labels", labelpath);
      find_replace(labelpath, "JPEGImages", "labels", labelpath);
      find_replace(labelpath, ".jpg", ".txt", labelpath);
      find_replace(labelpath, ".JPEG", ".txt", labelpath);
      
      int num_truth = 0;
      box_label *truth = read_boxes(labelpath, &num_truth);
      box *truthbox = (box *) malloc(sizeof(box) * num_truth);
      for (k = 0; k < l.w*l.h*l.n; ++k) {
	if(probs[k][0] > thresh) {
	  ++proposals;
	}
      }

      // iterate over ground truth boxes...
      
      for (j = 0; j < num_truth; ++j) {
	++total;
	box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
	truthbox[j] = t;
	float best_iou = 0.0;
	int best_iou_k = -1;
	// iterate over proposals
	
	for (k = 0; k < l.w*l.h*l.n; ++k) {
	  float iou = box_iou(boxes[k], t);
	  if (probs[k][0] > thresh && iou > best_iou) {
	    best_iou = iou;
	    best_iou_k = k;
	  }
	}

	avg_iou += best_iou;
	if (best_iou > iou_thresh) {
	  ++correct;
	}
      }

      if (do_draw) {
	printf("%i x %i\n", orig.w, orig.h);
	my_draw_detections(orig, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes, truthbox, num_truth, iou_thresh, 3);

	//	my_show_image_cv(orig, "predictions", 1);
	my_show_image_cv(orig, "predictions", 0);
	
      }

      if (do_draw)
	fprintf(stderr, "%4d %4d %4d %4d  RPs/Img: %.2f\tIOU: %.2f%% R %.2f%% P %.2f%% [thresh = %.05f]\n",
		i, correct, total, proposals,
		(float)proposals/(i+1), avg_iou*100/total,
		100.*correct/total,                         // recall
		100.*(float)correct/(float)proposals,       // precision
		thresh);

      free_image(orig);
      free_image(sized);
      free(truthbox);
      //      free(boxes);
      //      free_ptrs((void **)probs, l.w*l.h*l.n);
      if (do_draw) {
#ifdef OPENCV
	//	cvWaitKey(5);   // when we are writing
      cvWaitKey(0);
      //      cvDestroyAllWindows();
#endif
      }
    }

    *precision = (float)correct/(float)proposals;
    *recall = (float)correct/(float)total;
    *num_proposals = proposals;
    *num_correct = correct;
    *num_total = total;
}

void my_m46_test_detector(char *datacfg, char *cfgfile, char *weightfile, float thresh, int do_draw, float *precision, float *recall, int *num_proposals, int *num_correct, int *num_total)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();

    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    if (do_draw)
      fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    //    list *plist = get_paths("data/voc.2007.test");
    list *plist = get_paths("data/m46_scallop_test.txt");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;

    int j, k;
    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

    /*    float thresh = .001; */
    float iou_thresh = 0;
    //    float iou_thresh = .5;
    float nms = .5 * .5; // do nms as normal

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;
    int wrong = 0;

    if (do_draw)
      printf("MY testing detector with thresh %f\n", thresh);

    char *outfilename = (char *) malloc(256 * sizeof(char));
    
    for (int i = 0; i < m; i++ ) {

      char *path = paths[i];

      if (do_draw)
	printf("%s\n", path);
      
      image orig = load_image_color(path, 0, 0);
      image sized = resize_image(orig, net.w, net.h);
      char *id = basecfg(path);
      network_predict(net, sized.data);
      get_region_boxes(l, 1, 1, thresh, probs, boxes, 1, 0, .5);
      if (nms)
	do_nms(boxes, probs, l.w*l.h*l.n, 1, nms);
      
      char labelpath[4096];
      find_replace(path, "images", "labels", labelpath);
      find_replace(labelpath, "JPEGImages", "labels", labelpath);
      find_replace(labelpath, ".jpg", ".txt", labelpath);
      find_replace(labelpath, ".JPEG", ".txt", labelpath);
      
      int num_truth = 0;
      box_label *truth = read_boxes(labelpath, &num_truth);
      box *truthbox = (box *) malloc(sizeof(box) * num_truth);
      for (k = 0; k < l.w*l.h*l.n; ++k) {
	if(probs[k][0] > thresh) {
	  ++proposals;
	}
      }

      // iterate over ground truth boxes...
      
      for (j = 0; j < num_truth; ++j) {
	++total;
	box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
	truthbox[j] = t;
	float best_iou = 0.0;
	int best_iou_k = -1;
	// iterate over proposals
	
	for (k = 0; k < l.w*l.h*l.n; ++k) {
	  float iou = box_iou(boxes[k], t);
	  if (probs[k][0] > thresh && iou > best_iou) {
	    best_iou = iou;
	    best_iou_k = k;
	  }
	}

	avg_iou += best_iou;
	if (best_iou > iou_thresh) {
	  ++correct;
	}
      }

      if (do_draw) {
	printf("%i x %i\n", orig.w, orig.h);
	my_draw_detections(orig, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes, truthbox, num_truth, iou_thresh, 3);

	my_show_image_cv(orig, "predictions", 1);
	//my_show_image_cv(orig, "predictions", 0);
	
      }

      if (do_draw)
	fprintf(stderr, "%4d %4d %4d %4d  RPs/Img: %.2f\tIOU: %.2f%% R %.2f%% P %.2f%% [thresh = %.05f]\n",
		i, correct, total, proposals,
		(float)proposals/(i+1), avg_iou*100/total,
		100.*correct/total,                         // recall
		100.*(float)correct/(float)proposals,       // precision
		thresh);

      free_image(orig);
      free_image(sized);
      free(truthbox);
      //      free(boxes);
      //      free_ptrs((void **)probs, l.w*l.h*l.n);
      if (do_draw) {
#ifdef OPENCV
	//	cvWaitKey(5);   // when we are writing
      cvWaitKey(0);
      //      cvDestroyAllWindows();
#endif
      }
    }

    *precision = (float)correct/(float)proposals;
    *recall = (float)correct/(float)total;
    *num_proposals = proposals;
    *num_correct = correct;
    *num_total = total;
}


// no drawing; network and weights already loaded

void my2_test_detector(network *netptr, char *test_filename, float thresh, float *precision, float *recall, int *num_proposals, int *num_correct, int *num_total)
{
    //    list *plist = get_paths("data/voc.2007.test");
    //    list *plist = get_paths("data/scallop_test.txt");
    list *plist = get_paths(test_filename);
    char **paths = (char **)list_to_array(plist);

    layer l = netptr->layers[netptr->n-1];
    int classes = l.classes;

    int j, k;
    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

    /*    float thresh = .001; */
    float iou_thresh = .5;
    float nms = .5 * iou_thresh;   // .4

    int total = 0;
    int correct = 0;
    int true_correct = 0;
    int proposals = 0;
    float avg_iou = 0;
    int wrong = 0;

    // initially all detections/proposals are false positives

    int num = l.w*l.h*l.n;
    int *true_pos = (int *) malloc(sizeof(int) * num);
    for (i = 0; i < num; i++)
      true_pos[i] = 0;

    // iterate over images
    
    for (int i = 0; i < m; i++ ) {

      char *path = paths[i];

      image orig = load_image_color(path, 0, 0);
      image sized = resize_image(orig, netptr->w, netptr->h);
      char *id = basecfg(path);
      network_predict(*netptr, sized.data);
      get_region_boxes(l, 1, 1, thresh, probs, boxes, 1, 0, .5);
      if (nms)
	do_nms(boxes, probs, l.w*l.h*l.n, 1, nms);
      
      char labelpath[4096];
      find_replace(path, "images", "labels", labelpath);
      find_replace(labelpath, "JPEGImages", "labels", labelpath);
      find_replace(labelpath, ".jpg", ".txt", labelpath);
      find_replace(labelpath, ".JPEG", ".txt", labelpath);
      
      int num_truth = 0;
      box_label *truth = read_boxes(labelpath, &num_truth);
      box *truthbox = (box *) malloc(sizeof(box) * num_truth);
      for (k = 0; k < l.w*l.h*l.n; ++k) {
	if(probs[k][0] > thresh) {
	  proposals++;
	}
      }

      // iterate over ground truth boxes...
      
      for (j = 0; j < num_truth; ++j) {
	total++;
	box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
	truthbox[j] = t;
	float best_iou = 0.0;
	int best_iou_k = -1;
	// iterate over proposals
	
	for (k = 0; k < l.w*l.h*l.n; ++k) {
	  float iou = box_iou(boxes[k], t);
	  if (probs[k][0] > thresh && iou > best_iou) {
	    best_iou = iou;
	    best_iou_k = k;
	  }
	}

	avg_iou += best_iou;
	if (best_iou > iou_thresh) {
	  true_pos[best_iou_k] = 1;
	  correct++;
	}
      }

      // to avoid double counting
      
      for (j = 0; j < num; j++) {
	if (true_pos[j] == 1)
	  true_correct++;
      }
      
      free_image(orig);
      free_image(sized);
      free(truthbox);
      //      free(boxes);
      //      free_ptrs((void **)probs, l.w*l.h*l.n);
    }

    //    printf("correct: old %i, new %i\n", correct, true_correct);

    if (correct > proposals)
      correct = proposals;

    //correct = true_correct;
    
    if (proposals == 0)
      *precision = 1.0;
    else
      *precision = (float)correct/(float)proposals;
    if (total == 0)
      *recall = 1.0;
    else
      *recall = (float)correct/(float)total;
    *num_proposals = proposals;
    *num_correct = correct;
    *num_total = total;
}

// no drawing; network and weights already loaded

void my2_m46_test_detector(network *netptr, char *test_filename, float thresh, float *precision, float *recall, int *num_proposals, int *num_correct, int *num_total)
{
    //    list *plist = get_paths("data/voc.2007.test");
    //    list *plist = get_paths("data/scallop_test.txt");
    list *plist = get_paths(test_filename);
    char **paths = (char **)list_to_array(plist);

    layer l = netptr->layers[netptr->n-1];
    int classes = l.classes;

    int j, k;
    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

    /*    float thresh = .001; */
    float iou_thresh = 0; // .5;
    float nms = .5 * .5; // iou_thresh;   // .4

    int total = 0;
    int correct = 0;
    int true_correct = 0;
    int proposals = 0;
    float avg_iou = 0;
    int wrong = 0;

    // initially all detections/proposals are false positives

    int num = l.w*l.h*l.n;
    int *true_pos = (int *) malloc(sizeof(int) * num);
    for (i = 0; i < num; i++)
      true_pos[i] = 0;

    // iterate over images
    
    //    for (int i = 0; i < m; i++ ) {
    for (int i = 0; i < 100; i++ ) {

      char *path = paths[i];

      //      printf("%i %s\n", i, path);
      
      image orig = load_image_color(path, 0, 0);
      image sized = resize_image(orig, netptr->w, netptr->h);
      char *id = basecfg(path);
      network_predict(*netptr, sized.data);
      get_region_boxes(l, 1, 1, thresh, probs, boxes, 1, 0, .5);
      if (nms)
	do_nms(boxes, probs, l.w*l.h*l.n, 1, nms);
      
      char labelpath[4096];
      find_replace(path, "images", "labels", labelpath);
      find_replace(labelpath, "JPEGImages", "labels", labelpath);
      find_replace(labelpath, ".jpg", ".txt", labelpath);
      find_replace(labelpath, ".JPEG", ".txt", labelpath);
      
      int num_truth = 0;
      box_label *truth = read_boxes(labelpath, &num_truth);
      box *truthbox = (box *) malloc(sizeof(box) * num_truth);
      for (k = 0; k < l.w*l.h*l.n; ++k) {
	if(probs[k][0] > thresh) {
	  proposals++;
	}
      }

      // iterate over ground truth boxes...
      
      for (j = 0; j < num_truth; ++j) {
	total++;
	box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
	truthbox[j] = t;
	float best_iou = 0.0;
	int best_iou_k = -1;
	// iterate over proposals
	
	for (k = 0; k < l.w*l.h*l.n; ++k) {
	  float iou = box_iou(boxes[k], t);
	  if (probs[k][0] > thresh && iou > best_iou) {
	    best_iou = iou;
	    best_iou_k = k;
	  }
	}

	avg_iou += best_iou;
	if (best_iou > iou_thresh) {
	  true_pos[best_iou_k] = 1;
	  correct++;
	}
      }

      // to avoid double counting
      
      for (j = 0; j < num; j++) {
	if (true_pos[j] == 1)
	  true_correct++;
      }
      
      free_image(orig);
      free_image(sized);
      free(truthbox);
      //      free(boxes);
      //      free_ptrs((void **)probs, l.w*l.h*l.n);
    }

    //    printf("correct: old %i, new %i\n", correct, true_correct);

    if (correct > proposals)
      correct = proposals;

    //correct = true_correct;
    
    if (proposals == 0)
      *precision = 1.0;
    else
      *precision = (float)correct/(float)proposals;
    if (total == 0)
      *recall = 1.0;
    else
      *recall = (float)correct/(float)total;
    *num_proposals = proposals;
    *num_correct = correct;
    *num_total = total;
}

void my_prcurve_detector(char *datacfg, char *cfgfile, char *weightfile)
{
  float precision, recall;
  int num_thresh = 19;
  int num_proposals, num_correct, num_total;
  float *thresh = (float *) malloc(sizeof(float)*num_thresh);

  thresh[0] = -1.0;
  thresh[1] = 0.0;
  thresh[2] = 0.001;
  thresh[3] = 0.01;
  thresh[4] = 0.1;
  thresh[5] = 0.2;
  thresh[6] = 0.3;
  thresh[7] = 0.4;
  thresh[8] = 0.5;
  thresh[9] = 0.6;
  thresh[10] = 0.7;
  thresh[11] = 0.75;
  thresh[12] = 0.8;
  thresh[13] = 0.825;
  thresh[14] = 0.85;
  thresh[15] = 0.875;
  thresh[16] = 0.9;
  thresh[17] = 0.925;
  thresh[18] = 0.95;

  // get network set up
  
  network net = parse_network_cfg(cfgfile);
  if (weightfile)
    load_weights(&net, weightfile);
  set_batch_network(&net, 1);
  srand(time(0));

  // go time

  //  char *test_filename = (char *) malloc(sizeof(char)*256);
  //  sprintf(test_filename, "%s", "data/scallop_test.txt");
  list *options = read_data_cfg(datacfg);
  char *test_filename = option_find_str(options, "valid", "data/scallop_test.list");

  printf("threshold, recall, precision, num proposals, num correct, num total\n");
  for (int i = 0; i < num_thresh; i++) {
    my2_test_detector(&net, test_filename, thresh[i], &precision, &recall, &num_proposals, &num_correct, &num_total);
    printf("%.3f, %.3f, %.3f, %i, %i, %i\n", thresh[i], recall, precision, num_proposals, num_correct, num_total);
  }  
}

void my_m46_prcurve_detector(char *datacfg, char *cfgfile, char *weightfile)
{
  float precision, recall;
  int num_thresh = 19;
  int num_proposals, num_correct, num_total;
  float *thresh = (float *) malloc(sizeof(float)*num_thresh);

  thresh[0] = -1.0;
  thresh[1] = 0.0;
  thresh[2] = 0.001;
  thresh[3] = 0.01;
  thresh[4] = 0.1;
  thresh[5] = 0.2;
  thresh[6] = 0.3;
  thresh[7] = 0.4;
  thresh[8] = 0.5;
  thresh[9] = 0.6;
  thresh[10] = 0.7;
  thresh[11] = 0.75;
  thresh[12] = 0.8;
  thresh[13] = 0.825;
  thresh[14] = 0.85;
  thresh[15] = 0.875;
  thresh[16] = 0.9;
  thresh[17] = 0.925;
  thresh[18] = 0.95;

  // get network set up
  
  network net = parse_network_cfg(cfgfile);
  if (weightfile)
    load_weights(&net, weightfile);
  set_batch_network(&net, 1);
  srand(time(0));

  // go time

  //  char *test_filename = (char *) malloc(sizeof(char)*256);
  //  sprintf(test_filename, "%s", "data/scallop_test.txt");
  list *options = read_data_cfg(datacfg);
  char *test_filename = option_find_str(options, "valid", "data/m46_scallop_test.list");

  printf("threshold, recall, precision, num proposals, num correct, num total\n");
  for (int i = 0; i < num_thresh; i++) {
    my2_m46_test_detector(&net, test_filename, thresh[i], &precision, &recall, &num_proposals, &num_correct, &num_total);
    printf("%.3f, %.3f, %.3f, %i, %i, %i\n", thresh[i], recall, precision, num_proposals, num_correct, num_total);
  }  
}

void run_detector(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    //    float thresh = find_float_arg(argc, argv, "-thresh", .24);
    float thresh = find_float_arg(argc, argv, "-thresh", .2);
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    float precision, recall;
    int num_proposals, num_correct, num_total;
    if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh);
    else if(0==strcmp(argv[2], "m46test")) my_m46_test_detector(datacfg, cfg, weights, thresh, 1, &precision, &recall, &num_proposals, &num_correct, &num_total);
    else if(0==strcmp(argv[2], "mytest")) my_test_detector(datacfg, cfg, weights, thresh, 1, &precision, &recall, &num_proposals, &num_correct, &num_total);
    else if(0==strcmp(argv[2], "mypr")) my_prcurve_detector(datacfg, cfg, weights);
    else if(0==strcmp(argv[2], "m46pr")) my_m46_prcurve_detector(datacfg, cfg, weights);
    else if(0==strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "recall")) validate_detector_recall(cfg, weights, thresh);
    else if(0==strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix, hier_thresh);
    }
}
