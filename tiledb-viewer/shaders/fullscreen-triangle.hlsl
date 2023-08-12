//T: fullscreen-triangle ps:PSMain vs:VSMain

struct GridInfo {
    int2 GridBegin;
};

struct ShaderUniforms {
  float2 Position;
  float2 Scale;
  uint2 ViewRange;
  uint2 GridSize;
  uint2 TileSize;
  uint NumGrids;
  
  float AspectRatio;
  float Gamma;
  float GridAmount;
};

[[vk::binding(0, 0)]] ConstantBuffer<ShaderUniforms> shaderUniforms;
[[vk::binding(1, 0)]] uniform StructuredBuffer<GridInfo> gridInfos;
[[vk::binding(2, 0)]] uniform StructuredBuffer<int> occupied;
[[vk::binding(3, 0)]] uniform Texture2DArray<uint> textureImage;

struct FragShaderInput {
  float4 position : SV_Position;
  float2 UV : TEXCOORD0;
};

bool GetGridToRead(int2 PixID, out int Res) {
  int SmallestValidRes = -1;
  for (Res = 0; Res < shaderUniforms.NumGrids; ++Res) {
    int2 LevelTileSize = int2(shaderUniforms.TileSize) << Res;
    if (!(
      all(PixID >= gridInfos[Res].GridBegin) &&
      all(PixID < gridInfos[Res].GridBegin + int2(shaderUniforms.GridSize * LevelTileSize))
    )) {
      continue;
    }
    
    SmallestValidRes = Res;

    int2 LocalGridCoord = (PixID / LevelTileSize) - (gridInfos[Res].GridBegin / LevelTileSize);
    uint OccupiedBeginIndex = Res * shaderUniforms.GridSize.x * shaderUniforms.GridSize.y;
    uint LocalOccupiedIndex = LocalGridCoord.y * shaderUniforms.GridSize.x + LocalGridCoord.x;
    if (occupied[OccupiedBeginIndex + LocalOccupiedIndex] == 0) {
      continue;
    }

    return true;
  }
  
  Res = SmallestValidRes;
  
  return false;
}

bool ReadTilePixel(int2 PixID, out uint Res, out int GridIndex) {
  if (GetGridToRead(PixID, GridIndex)) {
    int2 AccessID = (PixID >> GridIndex) % int2(shaderUniforms.GridSize * shaderUniforms.TileSize);
    Res = textureImage[int3(AccessID, GridIndex)];
    return true;
  }
  return false;
}

float4 PixelColor(uint Value) {
    Value = clamp(Value, shaderUniforms.ViewRange.x, shaderUniforms.ViewRange.y);
    float sampledData = (Value - shaderUniforms.ViewRange.x) / float(shaderUniforms.ViewRange.y - shaderUniforms.ViewRange.x);
    sampledData = pow(sampledData, shaderUniforms.Gamma);
    return float4(sampledData, sampledData, sampledData, 1.0);
}

float4 PSMain(FragShaderInput vertexAttribs) : SV_TARGET {
  vertexAttribs.UV += float2(-0.5, -0.5);
  vertexAttribs.UV *= shaderUniforms.Scale;
  vertexAttribs.UV += shaderUniforms.Position;
  
  uint Value;
  int GridIndex;
  float4 PixelColorValue = float4(0.0, 0.0, 0.0, 1.0);
  if (ReadTilePixel(vertexAttribs.UV * shaderUniforms.TileSize, Value, GridIndex)) {
      PixelColorValue = PixelColor(Value);
  }
  
  float4 GridColor = (GridIndex == -1) ? float4(0, 0, 0, 1) : float4((4 - GridIndex) * 0.25, 0, (4 - GridIndex) * 0.25, 1);

  return lerp(PixelColorValue, GridColor, shaderUniforms.GridAmount);
}

FragShaderInput VSMain(uint vertexId : SV_VertexID) {
    FragShaderInput res = {
        float4(4 * (vertexId == 1), -4 * (vertexId == 2), 0, 1.0) + float4(-1, 1, 0, 0),
        float2(2 * (vertexId == 1), 2 * (vertexId == 2))
    };
    return res;
}
