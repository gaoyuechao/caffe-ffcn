// ------------------------------------------------------------------
// PR-FCN
// Licensed under The MIT License [see pr-fcn/LICENSE for details]
// Written by Yi Li
// Modified by Gao Yuechao
// ------------------------------------------------------------------

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/prfcn_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {
  template <typename Dtype>
  void PPSROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
    PPSROIPoolingParameter ppsroi_pooling_param = this->layer_param_.ppsroi_pooling_param();
    spatial_scale_ = ppsroi_pooling_param.spatial_scale();
    LOG(INFO) << "Spatial scale: " << spatial_scale_;

    CHECK_GT(ppsroi_pooling_param.output_dim(), 0)
      << "output_dim must be > 0";
    CHECK_GT(ppsroi_pooling_param.part_size(), 0)
      << "part_size must be > 0";

    output_dim_ = ppsroi_pooling_param.output_dim();
    part_size_ = ppsroi_pooling_param.part_size();
    pooled_height_ = part_size_;
    pooled_width_ = part_size_;
  }

  template <typename Dtype>
  void PPSROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    channels_ = bottom[0]->channels();
    CHECK_EQ(channels_, output_dim_*part_size_*part_size_)
      << "input channel number does not match layer parameters";
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    top[0]->Reshape(bottom[1]->num(), output_dim_*part_size_*part_size_, pooled_height_, pooled_width_);
    mapping_channel_.Reshape(bottom[1]->num(), output_dim_*part_size_*part_size_, pooled_height_, pooled_width_);
    max_loc_.Reshape(bottom[1]->num(), output_dim_*part_size_*part_size_, pooled_height_, pooled_width_);
  }

  template <typename Dtype>
  void PPSROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
    NOT_IMPLEMENTED;
  }

  template <typename Dtype>
  void PPSROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
    NOT_IMPLEMENTED;
  }
#ifdef CPU_ONLY
  STUB_GPU(PPSROIPoolingLayer);
#endif

  INSTANTIATE_CLASS(PPSROIPoolingLayer);
  REGISTER_LAYER_CLASS(PPSROIPooling);

}