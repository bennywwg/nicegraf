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
#include "shader-loader.h"

#include "imgui.h"

#include "tiledb-util.hpp"

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

#include <Database/DatabaseClient.hpp>
#include <Commands/AllCommands.hpp>
#include <Transfer/ImageTransfer.hpp>
#include <glm/ext.hpp>

#define TEST_MODE false

#define MAX_NUM_GRIDS 4

template<typename T>
T CorrectMod(T Lhs, T Rhs) {
    return (Rhs + (Lhs % Rhs)) % Rhs;
}

namespace ngf_samples {

ngf_image_format format_tiledb_to_ngf(ImageFormat Format) {
    if (Format.Type.Type == NumberType::U16) {
        if (Format.Type.NumChannels == ChannelCount::One) return NGF_IMAGE_FORMAT_R16U;
    } else if (Format.Type.Type == NumberType::I16) {
        if (Format.Type.NumChannels == ChannelCount::One) return NGF_IMAGE_FORMAT_R16S;
    }
    return NGF_IMAGE_FORMAT_UNDEFINED;
}

AutoReflect::EncodedImage MakeFakeImage(glm::ivec3 Coord) {
    auto Format = AutoReflect::ImageFormat {
        .Type = {
            .Type = NumberType::U16,
            .NumChannels = ChannelCount::One
        },
        .Size = glm::uvec3(1201, 1201, 1)
    };
    
    glm::ivec2 BasePos = glm::ivec2(Coord) * (1 << Coord.z) * glm::ivec2(1201, 1201);
    
    std::vector<uint8_t> ImgData(1201 * 1201 * 2);
    for (size_t i = 0; i < ImgData.size(); i += 2) {
        glm::ivec2 FullPos = BasePos + glm::ivec2((i/2) % 1201, (i/2) / 1201) * (1 << Coord.z);
        float Val = glm::sin(glm::length(glm::vec2(FullPos)) * 0.1f) + 1.0f;
        *reinterpret_cast<uint16_t*>(&ImgData[i]) = uint16_t(Val * 5000);
    }
    return AutoReflect::EncodedImage {
        .Format = {
            .Format = Format
        },
        .Data = ImgData
    };
}

struct ui_state {
    std::string ServerURI = "127.0.0.1";
    
    int NumGrids = MAX_NUM_GRIDS;
    
    float X = 100.0f;
    float Y = 111.0f;
    float Zoom = 0.0f;
    
    glm::ivec2 ViewRange = glm::ivec2(0, 10000);
    float Gamma = 1.0f;
    
    bool ShowGrids = false;
    bool FreezeGridLevels = true;
    bool FreezeGridLocations = false;
};

struct GPU_Data {
    struct GridInfo {
        glm::ivec2 GridBegin;
    };
    
    struct ShaderUniforms {
        glm::vec2 Position;
        glm::vec2 Scale;
        glm::uvec2 ViewRange;
        glm::uvec2 GridSize;
        glm::uvec2 TileSize;
        uint32_t NumGrids;
        
        float AspectRatio;
        float Gamma;
        float GridAmount;
    };
    
    AutoReflect::ImageFormat Format;
    glm::uvec2   GridSize;
    uint32_t     NumGrids;
    std::vector<GridInfo> GridInfos;
    // List of tiles that are present on the data texture
    std::vector<glm::ivec3> GPUPopulatedTiles;
    
    ngf::graphics_pipeline pipeline;
    ngf::buffer StagingBuffer; // For image data, need to rewrite
    
    ngf::uniform_multibuffer<ShaderUniforms> UniformMultibuffer;
    storage_multibuffer<GridInfo> GridInfosMultibuf;
    storage_multibuffer<int32_t> OccupiedMultibuf;
    image_uploader ImageUploader;
    ngf::image ImageData;
    
    glm::ivec2 GetGridLocationFromViewLocation(glm::vec2 ViewCenter, uint32_t GridLevel) {
        
        glm::ivec2 BaseTileSizePixels = Format.Size;
        glm::ivec2 LevelTileSizePixels = glm::ivec2(BaseTileSizePixels << static_cast<int32_t>(GridLevel));
        
        ViewCenter -= (glm::vec2(GridSize << GridLevel) * 0.5f - 0.5f * (1 << GridLevel));
        
        glm::ivec2 imv = glm::ivec2(ViewCenter * glm::vec2(BaseTileSizePixels));
        
        return LevelTileSizePixels * (imv / LevelTileSizePixels);
    }
    
    void UpdateGridInfos(glm::vec2 ViewCenter) {
        GridInfos.clear();
        for (uint32_t i = 0; i < NumGrids; ++i) {
            glm::ivec2 Begin =  GetGridLocationFromViewLocation(ViewCenter, i);
            GridInfos.push_back(GPU_Data::GridInfo { .GridBegin = Begin } );
        }
    }
    
    std::vector<glm::ivec3> GetTileIDsInGrids() {
        std::vector<glm::ivec3> Res;
        
        for (size_t i = 0; i < GridInfos.size(); ++i) {
            const GridInfo& Grid = GridInfos[i];
            for (uint32_t x = 0; x < GridSize.x; ++x) {
                const int32_t TileX = Grid.GridBegin.x / static_cast<int32_t>(Format.Size.x << i) + static_cast<int32_t>(x);
                for (uint32_t y = 0; y < GridSize.y; ++y) {
                    const int32_t TileY = Grid.GridBegin.y / static_cast<int32_t>(Format.Size.y << i) + static_cast<int32_t>(y);
                    Res.emplace_back(TileX, TileY, static_cast<int32_t>(i));
                }
            }
        }
        
        return Res;
    }
    
    // Return an occupied table for all the given tile ids, based off
    // The current grid location. Any tile ids outside of the current
    // grids are ignored
    std::vector<int32_t> GetOccupiedData(std::vector<glm::ivec3> const& TileIDs) {
        const size_t ElementsPerGrid = GridSize.x * GridSize.y;
        std::vector<int32_t> Res(NumGrids * ElementsPerGrid, 0);
        
        for (glm::ivec3 const& TileID : TileIDs) {
            if (TileID.z < 0 || TileID.z >= NumGrids) continue;
            
            const GridInfo& Grid = GridInfos[static_cast<size_t>(TileID.z)];
            
            glm::ivec2 GridBeginGridSpace = Grid.GridBegin / (glm::ivec2(Format.Size) << TileID.z);
            
            glm::ivec2 RelativeToGridSpace = glm::ivec2(TileID) - GridBeginGridSpace;
            
            if (RelativeToGridSpace.x < 0 || RelativeToGridSpace.x >= GridSize.x) continue;
            if (RelativeToGridSpace.y < 0 || RelativeToGridSpace.y >= GridSize.y) continue;
            
            Res[static_cast<size_t>(static_cast<int32_t>(ElementsPerGrid) * TileID.z + RelativeToGridSpace.y * static_cast<int32_t>(GridSize.x) + RelativeToGridSpace.x)] = 1;
        }
        
        return Res;
    }
    

    // Grid size required so that the grid covers the entire view, when the pixel density is at least PixelDensity
    // Pixel density is number of tile pixels per view pixel
    // Assumes square pixels
    // BufferPixels is how much buffer space added in addition to the regular viewport, to allow translation before new tile is needed
    glm::uvec2 CalculateRequiredGridSize(glm::uvec2 ViewSize, glm::uvec2 TileSize, uint32_t BufferPixels, float PixelDensity) {
        return glm::ceil(
            glm::vec2(ViewSize + TileSize - glm::uvec2(1) + BufferPixels) * PixelDensity / glm::vec2(TileSize)
        );
    }
    
    void Initialize(ngf_xfer_encoder xfer_encoder, AutoReflect::ImageFormat Format, glm::uvec2 GridSize, uint32_t NumGrids) {
        this->GridSize = GridSize;
        this->NumGrids = NumGrids;
        this->Format = Format;
        ImageData.initialize(ngf_image_info {
            .type = NGF_IMAGE_TYPE_IMAGE_2D,
            .extent = ngf_extent3d { Format.Size.x * GridSize.x, Format.Size.y * GridSize.y, 1 },
            .nmips = 1,
            .nlayers = MAX_NUM_GRIDS, // max num grids
            .format = format_tiledb_to_ngf(Format),
            .sample_count = NGF_SAMPLE_COUNT_1,
            .usage_hint = NGF_IMAGE_USAGE_XFER_DST | NGF_IMAGE_USAGE_STORAGE
        });
        
        StagingBuffer.initialize(ngf_buffer_info {
            .size = ImageUtils::ImageSize(Format),
            .storage_type = NGF_BUFFER_STORAGE_HOST_WRITEABLE,
            .buffer_usage = NGF_BUFFER_USAGE_XFER_SRC
        });
        
        UniformMultibuffer.initialize(3);
        
        GridInfosMultibuf.initialize_n(3, MAX_NUM_GRIDS);
        OccupiedMultibuf.initialize_n(3, 1024);
        
        ImageUploader.initialize(3, 4 * 1024 * 1024);
    }
};

struct viewer_state {
    GPU_Data GPUData;
    
    ui_state UI;
    
    DatabaseClient* client = nullptr;
    std::string TilesetUUID;
    std::vector<glm::ivec3> AllClientTiles;
    
    static constexpr size_t MaxImages = 2;
    std::mutex RequestMut;
    std::vector<glm::ivec3> RequestedCoords;
    std::mutex ResultsMut;
    std::vector<std::pair<AutoReflect::EncodedImage, glm::ivec3>> ResultImages;
    std::thread ClientThread;
    
    std::atomic_bool CloseClientThread = false;
    
    void SetRequestedCoords(const std::vector<glm::ivec3>& Requests) {
        std::lock_guard<std::mutex> RequestLock(RequestMut);
        RequestedCoords = Requests;
    }
    
    std::optional<std::pair<EncodedImage, glm::ivec3>> PopResultImage() {
        std::lock_guard<std::mutex> RequestLock(ResultsMut);
        if (ResultImages.size() > 0) {
            auto Res = ResultImages[0];
            ResultImages.erase(ResultImages.begin());
            return Res;
        } else {
            return std::nullopt;
        }
    }
    
    void ProcessClient() {
        while (!CloseClientThread) {
            glm::ivec3 ToGet;
            bool ShouldContinue = false;
            {
                std::lock_guard<std::mutex> RequestLock(RequestMut);
                if (RequestedCoords.size() == 0) {
                    ShouldContinue = true;
                } else {
                    ToGet = RequestedCoords[0];
                    RequestedCoords.erase(RequestedCoords.begin());
                }
            }
            
            if (!ShouldContinue) {
                std::lock_guard<std::mutex> RequestLock(ResultsMut);
                if (ResultImages.size() >= MaxImages) {
                    ShouldContinue = true;
                }
            }
            
            if (ShouldContinue) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            
            AutoReflect::CommandReadTile readCmd;
            readCmd.UUID.Coord.Coord = ToGet;
            readCmd.UUID.TilesetUUID = TilesetUUID;
            std::cout << "Requesting " << ToGet.x << ", " << ToGet.y << "\n";
            const auto now = std::chrono::system_clock::now();
            auto TileData = client->RequestSynchronous(readCmd);
            // Invert image along x and y axis
            if (TileData.Success) {
                uint16_t* Data = reinterpret_cast<uint16_t*>(&TileData.Image.Data[0]);
                for (size_t y = 0; y < TileData.Image.Format.Format.Size.y; ++y) {
                    for (size_t x = 0; x < TileData.Image.Format.Format.Size.x; ++x) {
                        size_t inv_x = TileData.Image.Format.Format.Size.x - x - 1;
                        if (inv_x <= x) break;
                        std::swap(Data[y * TileData.Image.Format.Format.Size.x + x], Data[y * TileData.Image.Format.Format.Size.x + inv_x]);
                    }
                }
                
                // Now y
                for (size_t x = 0; x < TileData.Image.Format.Format.Size.x; ++x) {
                    for (size_t y = 0; y < TileData.Image.Format.Format.Size.y; ++y) {
                        size_t inv_y = TileData.Image.Format.Format.Size.y - y - 1;
                        if (inv_y <= y) break;
                        std::swap(Data[y * TileData.Image.Format.Format.Size.x + x], Data[inv_y * TileData.Image.Format.Format.Size.x + x]);
                    }
                }

                // Swap endian
                for (size_t i = 0; i < TileData.Image.Data.size(); i += 2) {
                    std::swap(TileData.Image.Data[i], TileData.Image.Data[i + 1]);
                }
            }
            
            std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - now).count() << "ms\n";
            
            if (TileData.Success) {
                std::lock_guard<std::mutex> RequestLock(ResultsMut);
                ResultImages.emplace_back(TileData.Image, ToGet);
            }
        }
    }
    
    
};

void* sample_initialize(
    uint32_t         initial_window_width,
    uint32_t         initial_window_height,
    ngf_sample_count main_render_target_sample_count,
    ngf_xfer_encoder xfer_encoder) {
    viewer_state* state = new viewer_state();
    GPU_Data& gpuState = state->GPUData;
    
#if !TEST_MODE
    state->client = new DatabaseClient("192.168.1.71");
    
    if (!state->client->IsConnected()) {
        delete state->client;
        state->client = nullptr;
        return reinterpret_cast<void*>(state);
    }
    
    auto AllTilesetsResponse = state->client->RequestSynchronous(AutoReflect::CommandGetAllTilesets());
    if (!AllTilesetsResponse.Tilesets.empty()) {
        gpuState.Initialize(xfer_encoder, AllTilesetsResponse.Tilesets[0].Format.Format, glm::uvec2(3, 2), 4);
    } else {
        return reinterpret_cast<void*>(state);
    }
    
    AutoReflect::CommandGetAllTiles cmd;
    cmd.TilesetUUID = AllTilesetsResponse.Tilesets[0].ID;
    auto AllTiles = state->client->RequestSynchronous(cmd);
    for (auto Tile : *AllTiles.Tiles) {
        state->AllClientTiles.push_back(Tile.Coord);
    }
    
    state->TilesetUUID = AllTilesetsResponse.Tilesets[0].ID;
    
    state->ClientThread = std::thread([state]() {
        state->ProcessClient();
    });
#else
    auto Img = MakeFakeImage(glm::ivec3(0, 0, 0));
    
    gpuState.Initialize(xfer_encoder, Img.Format.Format, glm::uvec2(6, 4), 4);
    
    gpuState.UploadImageData(xfer_encoder, 0, glm::ivec2(0, 0), Img);
#endif

    const ngf::shader_stage vertex_shader_stage   = load_shader_stage("fullscreen-triangle", "VSMain", NGF_STAGE_VERTEX);
    const ngf::shader_stage fragment_shader_stage = load_shader_stage("fullscreen-triangle", "PSMain", NGF_STAGE_FRAGMENT);

    ngf_util_graphics_pipeline_data pipeline_data;
    ngf_util_create_default_graphics_pipeline_data(&pipeline_data);

    pipeline_data.pipeline_info.nshader_stages   = 2;
    pipeline_data.pipeline_info.shader_stages[0] = vertex_shader_stage.get();
    pipeline_data.pipeline_info.shader_stages[1] = fragment_shader_stage.get();

    pipeline_data.multisample_info.sample_count = main_render_target_sample_count;

    pipeline_data.pipeline_info.compatible_rt_attachment_descs = ngf_default_render_target_attachment_descs();

    gpuState.pipeline.initialize(pipeline_data.pipeline_info);
    
    return reinterpret_cast<void*>(state);
}

void sample_pre_draw_frame(ngf_cmd_buffer cmd_buffer, main_render_pass_sync_info* sync_op, void* userdata)
{
    auto state = reinterpret_cast<viewer_state*>(userdata);
    GPU_Data& gpuData = state->GPUData;
    gpuData.NumGrids = static_cast<uint32_t>(state->UI.NumGrids);
    
    if (!state->UI.FreezeGridLocations) {
        gpuData.UpdateGridInfos(glm::vec2(state->UI.X, state->UI.Y));
    }
    
    const std::vector<glm::ivec3> AllInGrid = gpuData.GetTileIDsInGrids();
    
    gpuData.GridInfosMultibuf.write_n(gpuData.GridInfos);
    
    // Debugging, add random grid to list
#if TEST_MODE
    {
        int randVal = rand();
        if (randVal >= 0) {
            randVal = randVal % static_cast<int32_t>(RequiredList.size());
            glm::ivec3 toAdd = RequiredList[(size_t)randVal];
            
            for(size_t i = 0; i < gpuData.GPUPopulatedTiles.size(); ++i) {
                if (gpuData.GPUPopulatedTiles[i] == toAdd) {
                    gpuData.GPUPopulatedTiles.erase(gpuData.GPUPopulatedTiles.begin() + (int64_t)i);
                    break;
                }
            }
            
            gpuData.GPUPopulatedTiles.push_back(toAdd);
            
            glm::ivec2 ModulodToAdd = CorrectMod(glm::ivec2(toAdd), glm::ivec2(gpuData.GridSize));
            
            gpuData.ImageUploader.update_section(MakeFakeImage(toAdd), glm::ivec3(
                                                                                  ModulodToAdd.x * static_cast<int32_t>(gpuData.Format.Size.x),
                                                                                  ModulodToAdd.y * static_cast<int32_t>(gpuData.Format.Size.y),
                                                                                  toAdd.z));
        }
    }
#else
    {
        std::vector<glm::ivec3> AllInGridNotLoaded = Difference(AllInGrid, gpuData.GPUPopulatedTiles);
        
        state->SetRequestedCoords(AllInGridNotLoaded);
        
        if (auto ImageOption = state->PopResultImage()) {
            if (Intersection({ ImageOption->second }, AllInGrid).size() > 0) {
                glm::ivec2 ModulodToAdd = CorrectMod(glm::ivec2(ImageOption->second), glm::ivec2(gpuData.GridSize));
                gpuData.ImageUploader.update_section(ImageOption->first, glm::ivec3(
                                                                                    ModulodToAdd.x * static_cast<int32_t>(gpuData.Format.Size.x),
                                                                                    ModulodToAdd.y * static_cast<int32_t>(gpuData.Format.Size.y),
                                                                                    ImageOption->second.z));
                
                gpuData.GPUPopulatedTiles.push_back(ImageOption->second);
            }
        }
    }
#endif
    
    gpuData.GPUPopulatedTiles = Intersection(AllInGrid, gpuData.GPUPopulatedTiles);
    
    gpuData.OccupiedMultibuf.write_n(gpuData.GetOccupiedData(gpuData.GPUPopulatedTiles));
    
    ngf_xfer_pass_info info {
        .sync_compute_resources = { }
    };
    
    ngf_xfer_encoder enc;
    ngf_cmd_begin_xfer_pass(cmd_buffer, &info, &enc);
    if (!state->UI.FreezeGridLocations) {
        gpuData.GridInfosMultibuf.enqueue_copy(enc);
        gpuData.GridInfosMultibuf.advance_frame();
    }
    gpuData.OccupiedMultibuf.enqueue_copy(enc);
    gpuData.OccupiedMultibuf.advance_frame();
    gpuData.ImageUploader.enqueue_copy(enc, gpuData.ImageData);
    gpuData.ImageUploader.advance_frame();
    ngf_cmd_end_xfer_pass(enc);
}

void sample_draw_frame(
    ngf_render_encoder main_render_pass,
    float              time_delta_ms,
    ngf_frame_token    frame_token,
    uint32_t           width,
    uint32_t           height,
    float              time,
    void*              userdata) {
    auto state = reinterpret_cast<viewer_state*>(userdata);
    GPU_Data& gpuData = state->GPUData;
    
    float AspectRatio = float(width) / float(height);

    gpuData.UniformMultibuffer.write(GPU_Data::ShaderUniforms {
        .Position = glm::vec2(state->UI.X, state->UI.Y),
        .Scale = glm::vec2(pow(2.0f, -state->UI.Zoom)) * glm::vec2(AspectRatio, 1.0f),
        .ViewRange = state->UI.ViewRange,
        .GridSize = gpuData.GridSize,
        .TileSize = gpuData.Format.Size,
        .NumGrids = gpuData.NumGrids,
        
        .AspectRatio = AspectRatio,
        .Gamma = state->UI.Gamma,
        .GridAmount = state->UI.ShowGrids ? 0.5f : 0.0f
    });
    
    ngf_cmd_bind_gfx_pipeline(main_render_pass, state->GPUData.pipeline.get());
    const ngf_irect2d viewport {0, 0, width, height};
    ngf_cmd_viewport(main_render_pass, &viewport);
    ngf_cmd_scissor(main_render_pass, &viewport);
    ngf::cmd_bind_resources(
                            main_render_pass,
                            gpuData.UniformMultibuffer.bind_op_at_current_offset(0, 0),
                            gpuData.GridInfosMultibuf.bind_op_at_current_offset(0, 1),
                            gpuData.OccupiedMultibuf.bind_op_at_current_offset(0, 2),
                            ngf::descriptor_set<0>::binding<3>::texture(state->GPUData.ImageData)
                            );
    ngf_cmd_draw(main_render_pass, false, 0, 3, 1);
}

void sample_post_submit(void* userdata)
{
    
}

void sample_post_draw_frame(ngf_cmd_buffer cmd_buffer, ngf_render_encoder prev_render_encoder, void* userdata)
{
    
}

void sample_draw_ui(void* userdata)
{
    viewer_state& state = *reinterpret_cast<viewer_state*>(userdata);
    
    const float MoveSpeed = glm::pow(2.0f, -state.UI.Zoom) * 0.25f;
    
#define FloatButtons(Name, Var, Amount)\
    ImGui::PushID(Name);\
    ImGui::InputFloat(Name, &Var, 0.0f, 0.0f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);\
    ImGui::SameLine();\
    if (ImGui::Button("-")) { Var -= Amount; }\
    ImGui::SameLine();\
    if (ImGui::Button("+")) { Var += Amount; }\
    ImGui::PopID();
    
    ImGui::Begin("TileDB", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse);
        
        ImGui::SliderInt("NumGrids", &state.UI.NumGrids, 1, MAX_NUM_GRIDS);
        FloatButtons("X", state.UI.X, MoveSpeed);
        FloatButtons("Y", state.UI.Y, MoveSpeed);
        FloatButtons("Zoom", state.UI.Zoom, 0.1f);
    
        ImGui::SliderInt("Min", &state.UI.ViewRange.x, 0, 10000 - 1);
        ImGui::SliderInt("Max", &state.UI.ViewRange.y, 0, 10000);
        ImGui::SliderFloat("Gamma", &state.UI.Gamma, 0.1f, 10.f);
        ImGui::Checkbox("Show Grid", &state.UI.ShowGrids);
        ImGui::Checkbox("Freeze Grid Levels", &state.UI.FreezeGridLevels);
        ImGui::Checkbox("Freeze Grid", &state.UI.FreezeGridLocations);
    ImGui::End();
}

void sample_shutdown(void* userdata) {
  auto data = static_cast<viewer_state*>(userdata);
    data->CloseClientThread = true;
    data->ClientThread.join();
  delete data;
  printf("shutting down\n");
}

}
