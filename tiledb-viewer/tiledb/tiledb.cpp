/**
 * Copyright (c) 2021 nicegraf contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "sample-interface.h"

#include "imgui.h"
#include <stdio.h>

#include <Database/DatabaseClient.hpp>

namespace ngf_samples {

struct sample_data {
  uint32_t magic_number = 0xdeadbeef;
    
  DatabaseClient* client = nullptr;
};

void* sample_initialize(
    uint32_t         initial_window_width,
    uint32_t         initial_window_height,
    ngf_sample_count main_render_target_sample_count,
    ngf_xfer_encoder xfer_encoder) {
  printf("sample initializing.\n");
  auto d = new sample_data{};
  d->client = new DatabaseClient("127.0.0.1");
  d->magic_number = 0xbadf00d;
  printf("sample initialization complete.\n");
  return static_cast<void*>(d);
}

void sample_pre_draw_frame(ngf_cmd_buffer cmd_buffer, main_render_pass_sync_info* sync_op, void* userdata)
{
    
}

void sample_draw_frame(
    ngf_render_encoder main_render_pass,
    float              time_delta_ms,
    ngf_frame_token    frame_token,
    uint32_t           width,
    uint32_t           height,
    float              time,
    void*              userdata) {
  auto data = static_cast<sample_data*>(userdata);
  //printf("drawing frame %d (w %d h %d) at time %f magic number 0x%x\n", (int)frame_token, width, height, time, data->magic_number);
}

void sample_post_submit(void* userdata)
{
    
}

void sample_post_draw_frame(ngf_cmd_buffer cmd_buffer, ngf_render_encoder prev_render_encoder, void* userdata)
{
    
}

void sample_draw_ui(void*)
{
    static float pitch = 0;
    static float yaw = 0;
    ImGui::Begin("Cubemap", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
      ImGui::SliderFloat("Pitch", &pitch, -1, 1);
      ImGui::SliderFloat("Yaw", &yaw, -1, 1);
      ImGui::Text("This sample uses textures by Emil Persson.\n"
                  "Licensed under CC BY 3.0\n"
                  "http://humus.name/index.php?page=Textures");
      ImGui::End();
}

void sample_shutdown(void* userdata) {
  auto data = static_cast<sample_data*>(userdata);
  delete data;
  printf("shutting down\n");
}

}
