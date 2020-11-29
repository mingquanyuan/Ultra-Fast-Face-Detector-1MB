// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the 
// specific language governing permissions and limitations under the License.

#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#include "tnn_sdk_sample.h"
#include "utils.h"
// #include "face_detector_struct.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace TNN_NS;



struct Pt{
    float _x;
    float _y;
};
struct bbox{
    float x1;
    float y1;
    float x2;
    float y2;
    float s;
    Pt pt[5];
};

struct box{
    float cx;
    float cy;
    float sx;
    float sy;
};

void create_anchor(std::vector<box> &anchor, int w, int h); 

void create_anchor_retinaface(std::vector<box> &anchor,  int cols, int rows);

void nms(std::vector<bbox> &input_boxes, float NMS_THRESH);

inline bool cmp(bbox a, bbox b) {
    if (a.s > b.s)
        return true;
    return false;
}


int main(int argc, char** argv) {
    if (argc < 3) {
        printf("how to run:  %s proto model height width\n", argv[0]);
        return -1;
    }
    // load模型文件
    auto proto_content = fdLoadFile(argv[1]);
    auto model_content = fdLoadFile(argv[2]);

    const float _threshold = 0.3;
    const float _nms = 0.4;

    // 设置图片尺寸
    int h = 240, w = 320;
    if(argc >= 5) {
        h = std::atoi(argv[3]);
        w = std::atoi(argv[4]);
    }
    // 读入图片
    char img_buff[256];
    char *input_imgfn = img_buff;
    if(argc < 6)
        strncpy(input_imgfn, "../../assets/test_face.jpg", 256);
    else
        strncpy(input_imgfn, argv[5], 256);
    printf("Face-detector is about to start, and the picrture is %s\n",input_imgfn);

    // use retinaface or not
    bool use_retinaface = true;
        
    int image_width, image_height, image_channel;
    unsigned char *im_bgr = stbi_load(input_imgfn, &image_width, &image_height, &image_channel, 3);
    
    bool resized = false;
    // resize image
    int target_w = 320;
    int target_h = 240;

    if (h != 240 && w !=320)
    {   
        resized = true;
        std::cout << "Need resize image..." << std::endl;
        unsigned char *im_bgr_resized;
        im_bgr_resized = (unsigned char*) malloc(target_w*target_h*image_channel);
        stbir_resize_uint8(im_bgr, image_width, image_height, 0, im_bgr_resized, target_w, target_h, 0, image_channel);
        // stbi_write_png("output_resized.png", target_w, target_h, image_channel, im_bgr_resized, 0);      // save the resized image 
        // sourcePixelscolor = (void*)im_bgr_resized;
        im_bgr = im_bgr_resized;
    }
    void* sourcePixelscolor = (void*)im_bgr;

    // 创建模型
    TNN_NS::TNN tnn;
    //创建模型配置
    TNN_NS::ModelConfig model_config;
    // 模型配置
    model_config.model_type = TNN_NS::MODEL_TYPE_TNN;
    model_config.params = {proto_content, model_content};
    // 配置初始化
    tnn.Init(model_config);

    // 创建网络配置MatConvertParam
    TNN_NS::NetworkConfig config;
    config.device_type = TNN_NS::DEVICE_ARM;
    TNN_NS::Status error;
    
    // 设置输入归一化参数
    TNN_NS::MatConvertParam input_cvt_param;
    input_cvt_param.scale = {1.0, 1.0, 1.0, 0.0};
    input_cvt_param.bias  = {-104.0, -117.0, -123.0, 0.0};

    //创建实例instance
    auto net_instance = tnn.CreateInst(config, error);
    
    // TNN_NS::DimsVector input_dim = {1,3, h, w};
    std::vector<int> nchw = {1, 3, target_h, target_w};
    std::shared_ptr<TNN_NS::Mat> input_mat(new TNN_NS::Mat(TNN_NS::DEVICE_ARM, TNN_NS::N8UC3, nchw, sourcePixelscolor));

    //设置实例的输入
    auto status = net_instance->SetInputMat(input_mat, input_cvt_param);

    // 实例前向推理
    Timer timer;
    timer.tic();
    status = net_instance->Forward();
    timer.toc("----total inference timer:");
    RETURN_ON_NEQ(status, TNN_NS::TNN_OK);

    // 声明Mat存放输出
    std::shared_ptr<TNN_NS::Mat> loc = nullptr;
    std::shared_ptr<TNN_NS::Mat> landms = nullptr;
    std::shared_ptr<TNN_NS::Mat> score = nullptr;

    if (use_retinaface) // use RetinaFace model
    {
        std::cout << "USE RetinaFace MODEL" << std::endl;
        status = net_instance->GetOutputMat(loc, MatConvertParam(), "loc");
        RETURN_ON_NEQ(status, TNN_NS::TNN_OK);
        status = net_instance->GetOutputMat(landms, MatConvertParam(), "landms");
        RETURN_ON_NEQ(status, TNN_NS::TNN_OK);
        status = net_instance->GetOutputMat(score, MatConvertParam(), "score");
        RETURN_ON_NEQ(status, TNN_NS::TNN_OK);
    }
    else // use RFB model
    {
        std::cout << "use RFB model" << std::endl;
        status = net_instance->GetOutputMat(loc, MatConvertParam(), "output0");
        RETURN_ON_NEQ(status, TNN_NS::TNN_OK);
        status = net_instance->GetOutputMat(landms, MatConvertParam(), "531");
        RETURN_ON_NEQ(status, TNN_NS::TNN_OK);
        status = net_instance->GetOutputMat(score, MatConvertParam(), "532");
        RETURN_ON_NEQ(status, TNN_NS::TNN_OK);
    }
    
    // 声明anchor存放priors
    std::vector<box> anchor;

    // use retinaface or not
    if (use_retinaface)
        create_anchor_retinaface(anchor,  target_w, target_h);
    else
        create_anchor(anchor,  target_w, target_h);

    std::cout << "anchor size: " << anchor.size() << std::endl;
    
    // 读取loc, score, landmarks数据
    float *loc_data = (float *)loc.get()->GetData();
    float *score_data = (float *)score.get()->GetData();
    float *landms_data = (float *)landms.get()->GetData();
    // std::cout << "loc: " << "channel: " << loc->GetChannel() << "|| height: " << loc->GetHeight() << "|| width: "<< loc->GetWidth() << std::endl;
    // std::cout << "score: " << "channel: " << score->GetChannel() << "|| height: " << score->GetHeight() << "|| width: "<< score->GetWidth() << std::endl;

    // 数据后处理
    std::vector<bbox > total_box;
    for (int i = 0; i < anchor.size(); ++i)
    {   
        int ind_score = 2 * i;
        int ind_loc = 4 * i;
        int ind_landms = 10 * i;
        
        if (score_data[ind_score + 1] > _threshold)
        {
            // std::cout << score_data[ind_score] <<" || " <<score_data[ind_score + 1] <<" || ind_score: " << ind_score << std::endl;
            box tmp = anchor[i];
            box tmp1;
            bbox result;

            // loc and score
            tmp1.cx = tmp.cx + loc_data[ind_loc] * 0.1 * tmp.sx;
            tmp1.cy = tmp.cy + loc_data[ind_loc + 1] * 0.1 * tmp.sy;
            tmp1.sx = tmp.sx * exp(loc_data[ind_loc + 2] * 0.2);
            tmp1.sy = tmp.sy * exp(loc_data[ind_loc + 3] * 0.2);

            result.x1 = (tmp1.cx - tmp1.sx/2) * target_w;
            if (result.x1<0)
                result.x1 = 0;
            result.y1 = (tmp1.cy - tmp1.sy/2) * target_h;
            if (result.y1<0)
                result.y1 = 0;
            result.x2 = (tmp1.cx + tmp1.sx/2) * target_w;
            if (result.x2>target_w)
                result.x2 = target_w;
            result.y2 = (tmp1.cy + tmp1.sy/2)* target_h;
            if (result.y2>target_h)
                result.y2 = target_h;
            result.s = score_data[ind_score + 1];

            // std::cout << "loc before conversion  : " <<  loc_data[ind_loc] << " | "  << loc_data[ind_loc + 1] << " | " << loc_data[ind_loc + 2] << " | " << loc_data[ind_loc + 3] << std::endl;
            // std::cout << "loc after conversion   : " <<  tmp1.cx << " | "  << tmp1.cy << " | " << tmp1.sx << " | " << tmp1.sy  << std::endl;
            // std::cout << "loc with img size ratio: " <<  result.x1 << " | "  << result.y1 << " | " << result.x2 << " | " << result.y2  << std::endl;
            // std::cout << "face score             : " << result.s << std::endl;

            // landmark
            for (int j = 0; j < 5; ++j)
            {
                result.pt[j]._x =( tmp.cx + landms_data[ind_loc + 2 * j] * 0.1 * tmp.sx ) * target_w;
                result.pt[j]._y =( tmp.cy + landms_data[ind_loc + 2 * j + 1] * 0.1 * tmp.sy ) * target_h;
            }

            total_box.push_back(result);
        }
    }

    std::cout << "total face detected in this image BEFORE nms: " << total_box.size() << std::endl<< std::endl;
    std::sort(total_box.begin(), total_box.end(), cmp);
    nms(total_box, _nms);
    std::cout << "total face detected in this image AFTER nms: " << total_box.size() << std::endl<< std::endl;

    // 脸部位置检测可视化
    uint8_t *ifm_buf = new uint8_t[target_w*target_h*4];
    for (int i = 0; i < target_w*target_h; ++i) {
        ifm_buf[i*4]   = im_bgr[i*3];
        ifm_buf[i*4+1] = im_bgr[i*3+1];
        ifm_buf[i*4+2] = im_bgr[i*3+2];
        ifm_buf[i*4+3] = 255;
    }
    for (int i = 0; i < total_box.size(); i++) {
        auto face = total_box[i];
        TNN_NS::Rectangle((void *)ifm_buf, target_h, target_w, face.x1, face.y1, face.x2,
                  face.y2, 1, 1);
    }

    int success = stbi_write_bmp("predictions.png", target_w, target_h, 4, ifm_buf);
    if(!success) 
        return -1;

    fprintf(stdout, "Face-detector done.\nNumber of faces: %d\n",int(total_box.size()));
    delete [] ifm_buf;

    free(im_bgr);
    std::cout << "Done..." << std::endl;
    return 0;
}


// retinaface
// compile and test on raspberry pi
// measure time usage for TNN and NCNN

void create_anchor(std::vector<box> &anchor, int w, int h)
{
//    anchor.reserve(num_boxes);
    anchor.clear();
    std::vector<std::vector<int> > feature_map(4), min_sizes(4);
    float steps[] = {8, 16, 32, 64};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h/steps[i]));
        feature_map[i].push_back(ceil(w/steps[i]));
    }
    std::vector<int> minsize1 = {10, 16, 24};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {32, 48};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {64, 96};
    min_sizes[2] = minsize3;
    std::vector<int> minsize4 = {128, 192, 256};
    min_sizes[3] = minsize4;


    for (int k = 0; k < feature_map.size(); ++k)
    {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for (int l = 0; l < min_size.size(); ++l)
                {
                    float s_kx = min_size[l]*1.0/w;
                    float s_ky = min_size[l]*1.0/h;
                    float cx = (j + 0.5) * steps[k]/w;
                    float cy = (i + 0.5) * steps[k]/h;
                    box axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }

    }

}


void create_anchor_retinaface(std::vector<box> &anchor, int w, int h)
{
//    anchor.reserve(num_boxes);
    anchor.clear();
    std::vector<std::vector<int> > feature_map(3), min_sizes(3);
    float steps[] = {8, 16, 32};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h/steps[i]));
        feature_map[i].push_back(ceil(w/steps[i]));
    }
    std::vector<int> minsize1 = {10, 20};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {32, 64};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {128, 256};
    min_sizes[2] = minsize3;

    for (int k = 0; k < feature_map.size(); ++k)
    {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for (int l = 0; l < min_size.size(); ++l)
                {
                    float s_kx = min_size[l]*1.0/w;
                    float s_ky = min_size[l]*1.0/h;
                    float cx = (j + 0.5) * steps[k]/w;
                    float cy = (i + 0.5) * steps[k]/h;
                    box axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }

    }

}


void nms(std::vector<bbox> &input_boxes, float NMS_THRESH)
{
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float   h = std::max(float(0), yy2 - yy1 + 1);
            float   inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}
