// ------------------------------------------------------------------
// PR-FCN
// Licensed under The MIT License [see pr-fcn/LICENSE for details]
// Written by Yi Li
// Modified by Gao Yuechao 2016-12
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/prfcn_layers.hpp"
#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {

  template <typename Dtype>
  __global__ void PPSROIPoolingForward(
    const int nthreads,
    const Dtype* bottom_data,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois,
    const int output_dim,
    const int part_size,
    Dtype* top_data,
    int* max_loc) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      //int ctop = (index / pooled_width / pooled_height) % output_dim;  //output_dim=output_dim_*part_size_*part_size_
      int n = index / pooled_width / pooled_height / output_dim; //number of ROI

      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
      Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
      Dtype roi_end_w = static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
      Dtype roi_end_h = static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom 
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      int hstart = floor(static_cast<Dtype>(ph) * bin_size_h + roi_start_h);
      int wstart = floor(static_cast<Dtype>(pw) * bin_size_w + roi_start_w);
      int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h + roi_start_h);
      int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w + roi_start_w);

      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0),width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      int gw = pw;
      int gh = ph;
      //int c = (ctop*part_size + gh)*part_size + gw;
      Dtype maxval = -FLT_MAX;
      int maxidx = -1;
      
      //bottom_data += (roi_batch_ind * channels + c) * height * width;
      bottom_data += (roi_batch_ind * channels) * height * width;

      Dtype out_sum = 0;
      for (int h = hstart; h < hend; ++h){
        for (int w = wstart; w < wend; ++w){
          int bottom_index = h*width + w;
          //out_sum += bottom_data[bottom_index];
          if (bottom_data[bottom_index] > maxval) {
            maxidx = bottom_index;
            maxval = bottom_data[bottom_index];
          }          
        }
      }

      //Dtype bin_area = (hend - hstart)*(wend - wstart);
      //top_data[index] = is_empty? 0. : out_sum/bin_area;
      top_data[index] = maxval;
      max_loc[index]=maxidx;
    }
  }

  template <typename Dtype>
  void PPSROIPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_rois = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int* max_loc_ptr = max_loc_.mutable_gpu_data();
    int count = top[0]->count();
    caffe_gpu_set(count, Dtype(0), top_data);
    caffe_gpu_set(count, -1, max_loc_ptr);
    // NOLINT_NEXT_LINE(whitespace/operators)
    PPSROIPoolingForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
      count, bottom_data, spatial_scale_, channels_, height_, width_, pooled_height_,
      pooled_width_, bottom_rois, output_dim_, part_size_, top_data, max_loc_ptr);
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  __global__ void PPSROIPoolingBackwardAtomic(
    const int nthreads,
    const Dtype* top_diff,
    const int* max_loc,
    const int num_rois,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int output_dim, 
    Dtype* bottom_diff,
    const Dtype* bottom_rois) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
      Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
      Dtype roi_end_w = static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
      Dtype roi_end_h = static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom 
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      int hstart = floor(static_cast<Dtype>(ph)* bin_size_h
        + roi_start_h);
      int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
        + roi_start_w);
      int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
        + roi_start_h);
      int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
        + roi_start_w);
      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0), width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      // Compute c at bottom
      Dtype* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels) * height * width;
      Dtype bin_area = (hend - hstart)*(wend - wstart);
      Dtype diff_val = is_empty ? 0. : top_diff[index] / bin_area;
      for (int h = hstart; h < hend; ++h){
        for (int w = wstart; w < wend; ++w){
          int bottom_index = h*width + w;
          if(max_loc[index]==bottom_index)
            caffe_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
        }
      }
    }
  }

  template <typename Dtype>
  void PPSROIPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
      return;
    }

    const Dtype* bottom_rois = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();
    const int* max_loc_ptr = max_loc_.gpu_data();
    caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
    caffe_gpu_set(bottom_count, Dtype(0), bottom_diff);
    const int count = top[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    PPSROIPoolingBackwardAtomic<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
      count, top_diff, max_loc_ptr, top[0]->num(), spatial_scale_,
      channels_, height_, width_, pooled_height_, pooled_width_, output_dim_,
      bottom_diff, bottom_rois);
    CUDA_POST_KERNEL_CHECK;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(PPSROIPoolingLayer);

}  // namespace caffe
