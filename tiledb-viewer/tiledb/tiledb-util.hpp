#include "sample-interface.h"

#include <vector>

#include <glm/ext.hpp>
#include <Transfer/ImageTransfer.hpp>

template<typename T> class storage_multibuffer {
  public:
    storage_multibuffer() = default;
    storage_multibuffer(storage_multibuffer&& other) {
    *this = std::move(other);
  }
    storage_multibuffer(const storage_multibuffer&) = delete;

    storage_multibuffer& operator=(storage_multibuffer&& other) = default;
    storage_multibuffer& operator=(const storage_multibuffer&)  = delete;

  ngf_error initialize_n(const uint32_t frames, const uint32_t num_elements) {
    const size_t alignment    = ngf_get_device_capabilities()->uniform_buffer_offset_alignment;
    const size_t aligned_size = ngf_util_align_size(sizeof(T) * num_elements, alignment);
    NGF_RETURN_IF_ERROR(buf_.initialize(ngf_buffer_info {
        aligned_size * frames,
        NGF_BUFFER_STORAGE_PRIVATE,
        NGF_BUFFER_USAGE_STORAGE_BUFFER | NGF_BUFFER_USAGE_XFER_DST}));
    NGF_RETURN_IF_ERROR(staging_buf_.initialize(ngf_buffer_info {
        aligned_size * frames,
        NGF_BUFFER_STORAGE_HOST_WRITEABLE,
        NGF_BUFFER_USAGE_XFER_SRC}));
    nframes_                = frames;
    aligned_per_frame_size_ = aligned_size;
    return NGF_ERROR_OK;
  }

  void write_n(const std::vector<T>& data) {
    current_offset_  = (frame_)*aligned_per_frame_size_;
    void* mapped_buf = ngf_buffer_map_range(staging_buf_.get(), current_offset_, aligned_per_frame_size_);
    memcpy(mapped_buf, (void*)data.data(), sizeof(T) * data.size());
    ngf_buffer_flush_range(staging_buf_.get(), 0, aligned_per_frame_size_);
    ngf_buffer_unmap(staging_buf_.get());
  }

  void enqueue_copy(ngf_xfer_encoder xfer_enc) {
    ngf_cmd_copy_buffer(xfer_enc, staging_buf_.get(), buf_.get(), aligned_per_frame_size_, current_offset_, current_offset_);
  }

  void advance_frame() {
    frame_ = (frame_ + 1u) % nframes_;
  }

  ngf_resource_bind_op bind_op_at_current_offset(
      uint32_t set,
      uint32_t binding,
      size_t   additional_offset = 0,
      size_t   range             = 0) const {
    ngf_resource_bind_op op;
    op.type               = NGF_DESCRIPTOR_STORAGE_BUFFER;
    op.target_binding     = binding;
    op.target_set         = set;
    op.info.buffer.buffer = buf_.get();
    op.info.buffer.offset = current_offset_ + additional_offset;
    op.info.buffer.range  = (range == 0) ? aligned_per_frame_size_ : range;
    return op;
  }

  private:
  ngf::buffer   buf_;
  ngf::buffer   staging_buf_;
  uint32_t frame_                  = 0;
  size_t   current_offset_         = 0;
  size_t   aligned_per_frame_size_ = 0;
  uint32_t nframes_                = 0;
};

class image_uploader {
  public:
    image_uploader() = default;
    image_uploader(image_uploader&& other) {
    *this = std::move(other);
  }
    image_uploader(const image_uploader&) = delete;

    image_uploader& operator=(image_uploader&& other) = default;
    image_uploader& operator=(const image_uploader&)  = delete;

  ngf_error initialize(const uint32_t frames, const uint32_t staging_buf_size) {
    NGF_RETURN_IF_ERROR(staging_buf_.initialize(ngf_buffer_info {
        staging_buf_size * frames,
        NGF_BUFFER_STORAGE_HOST_WRITEABLE,
        NGF_BUFFER_USAGE_XFER_SRC}));
    nframes_                = frames;
    size_per_frame_ = staging_buf_size;
    return NGF_ERROR_OK;
  }

  // Copy the data to staging buffer immediately, and add a pending write to the list
  // that will be copied to image with xfer encoder
  void update_section(const AutoReflect::EncodedImage& image, glm::uvec3 offset) {
      write_info write {
          .offset = offset,
          .extent = glm::uvec2(image.Format.Format.Size.x, image.Format.Format.Size.y),
          .staging_begin = current_offset_,
          .staging_size = image.Data.size()
      };
      current_offset_ += write.staging_size;
      current_frame_size_ += write.staging_size;
      
      if (current_frame_size_ > size_per_frame_) throw std::runtime_error("image_uploader ran out of data, need to increase staging buf size");
      
      pending_writes_.push_back(write);
      
    void* mapped_buf = ngf_buffer_map_range(staging_buf_.get(), write.staging_begin, write.staging_size);
    memcpy(mapped_buf, (void*)image.Data.data(), image.Data.size());
    ngf_buffer_flush_range(staging_buf_.get(), 0, write.staging_size);
    ngf_buffer_unmap(staging_buf_.get());
  }

  void enqueue_copy(ngf_xfer_encoder xfer_enc, const ngf::image& img) {
      for (const write_info& write : pending_writes_) {
          ngf_cmd_write_image(xfer_enc, staging_buf_, write.staging_begin, ngf_image_ref {
              .image = img,
              .mip_level = 0,
              .layer = write.offset.z
          }, ngf_offset3d {
              .x = static_cast<int32_t>(write.offset.x),
              .y = static_cast<int32_t>(write.offset.y),
              .z = static_cast<int32_t>(write.offset.z)
          }, ngf_extent3d {
              .width = write.extent.x,
              .height = write.extent.y,
              .depth = 1
          }, 1);
      }
      
      pending_writes_.clear();
  }

  void advance_frame() {
    frame_ = (frame_ + 1u) % nframes_;
     current_offset_ = frame_ * size_per_frame_;
      current_frame_size_ = 0;
  }

  private:
    struct write_info {
        glm::uvec3 offset;
        glm::uvec2 extent;
        size_t staging_begin;
        size_t staging_size;
    };
    
  ngf::buffer             staging_buf_;
  std::vector<write_info> pending_writes_;
  uint32_t frame_                  = 0;
  size_t   current_offset_         = 0;
  size_t   current_frame_size_     = 0;
  size_t   size_per_frame_         = 0;
  uint32_t nframes_                = 0;
};
