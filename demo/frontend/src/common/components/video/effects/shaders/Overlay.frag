#version 300 es
// Copyright (c) Meta Platforms, Inc. and affiliates.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

precision highp float;

in vec2 vTexCoord;

uniform sampler2D uSampler;
uniform vec2 uSize;
uniform int uNumMasks;
uniform float uOpacity;
uniform bool uBorder;

// Define texture samplers (WebGL typically has at least 8 texture units)
uniform sampler2D uMaskTexture0;
uniform sampler2D uMaskTexture1;
uniform sampler2D uMaskTexture2;
uniform sampler2D uMaskTexture3;
uniform sampler2D uMaskTexture4;
uniform sampler2D uMaskTexture5;
uniform sampler2D uMaskTexture6;
uniform sampler2D uMaskTexture7;
uniform sampler2D uMaskTexture8;
uniform sampler2D uMaskTexture9;
uniform sampler2D uMaskTexture10;
uniform sampler2D uMaskTexture11;
uniform sampler2D uMaskTexture12;
uniform sampler2D uMaskTexture13;
uniform sampler2D uMaskTexture14;
uniform sampler2D uMaskTexture15;

// Define color uniforms
uniform vec4 uMaskColor0;
uniform vec4 uMaskColor1;
uniform vec4 uMaskColor2;
uniform vec4 uMaskColor3;
uniform vec4 uMaskColor4;
uniform vec4 uMaskColor5;
uniform vec4 uMaskColor6;
uniform vec4 uMaskColor7;
uniform vec4 uMaskColor8;
uniform vec4 uMaskColor9;
uniform vec4 uMaskColor10;
uniform vec4 uMaskColor11;
uniform vec4 uMaskColor12;
uniform vec4 uMaskColor13;
uniform vec4 uMaskColor14;
uniform vec4 uMaskColor15;


uniform float uTime;
uniform vec2 uClickPos;
uniform int uActiveMask;

out vec4 fragColor;

vec4 lowerSaturation(vec4 color, float saturationFactor) {
  float luminance = 0.299f * color.r + 0.587f * color.g + 0.114f * color.b;
  vec3 gray = vec3(luminance);
  vec3 saturated = mix(gray, color.rgb, saturationFactor);
  return vec4(saturated, color.a);
}

vec4 detectEdges(sampler2D textureSampler, float coverage, vec4 edgeColor) {
  vec2 tvTexCoord = vec2(vTexCoord.y, vTexCoord.x);
  vec2 texOffset = 1.0f / uSize;
  vec3 result = vec3(0.0f);
  
  vec3 tLeft = texture(textureSampler, tvTexCoord + texOffset * vec2(-coverage, coverage)).rgb;
  vec3 tRight = texture(textureSampler, tvTexCoord + texOffset * vec2(coverage, -coverage)).rgb;
  vec3 bLeft = texture(textureSampler, tvTexCoord + texOffset * vec2(-coverage, -coverage)).rgb;
  vec3 bRight = texture(textureSampler, tvTexCoord + texOffset * vec2(coverage, coverage)).rgb;

  vec3 xEdge = tLeft + 2.0f * texture(textureSampler, tvTexCoord + texOffset * vec2(-coverage, 0)).rgb + bLeft - tRight - 2.0f * texture(textureSampler, tvTexCoord + texOffset * vec2(coverage, 0)).rgb - bRight;
  vec3 yEdge = tLeft + 2.0f * texture(textureSampler, tvTexCoord + texOffset * vec2(0, coverage)).rgb + tRight - bLeft - 2.0f * texture(textureSampler, tvTexCoord + texOffset * vec2(0, -coverage)).rgb - bRight;

  result = sqrt(xEdge * xEdge + yEdge * yEdge);
  return result.r > 1e-6f ? edgeColor : vec4(0.0f, 0.0f, 0.0f, 0.0f);
}

vec2 calculateAdjustedTexCoord(vec2 vTexCoord, vec4 bbox, float aspectRatio) {
  vec2 center = vec2((bbox.x + bbox.z) * 0.5f, bbox.w);
  float radiusX = abs(bbox.z - bbox.x);
  float radiusY = radiusX / aspectRatio;
  float scale = 1.0f;
  radiusX *= scale;
  radiusY *= scale;
  vec2 adjustedTexCoord = (vTexCoord - center) / vec2(radiusX, radiusY) + vec2(0.5f);
  return adjustedTexCoord;
}


void processMask(int maskIndex, sampler2D maskTexture, vec4 maskColor, inout vec4 finalColor, inout float totalMaskValue, inout vec4 edgeColor) {
  vec2 tvTexCoord = vec2(vTexCoord.y, vTexCoord.x);
  float maskValue = texture(maskTexture, tvTexCoord).r;
  float saturationFactor = 0.7;
  float timeThreshold = 1.1;
  float numRipples = 1.75;

  vec4 saturatedColor = lowerSaturation(maskColor / 255.0, saturationFactor);
  vec4 plainColor = vec4(vec3(saturatedColor).rgb, 1.0);
  vec4 rippleColor = vec4(maskColor.rgb / 255.0, 0.2);

  if (uActiveMask == maskIndex && uTime < timeThreshold) {
    float dist = length(vTexCoord - uClickPos);
    float colorFactor = abs(sin((dist - uTime) * numRipples));
    plainColor = vec4(mix(rippleColor.rgb, plainColor.rgb, colorFactor), 1.0);
  }

  finalColor += maskValue * plainColor;
  totalMaskValue += maskValue;

  if (edgeColor.a <= 0.0) {
    edgeColor = detectEdges(maskTexture, 1.25, maskColor / 255.0);
  }
}


void main() {
  vec4 color = texture(uSampler, vTexCoord);
  float timeThreshold = 1.1;
  float aspectRatio = uSize.y / uSize.x;
  vec2 tvTexCoord = vec2(vTexCoord.y, vTexCoord.x);
  vec2 adjustedClickCoord = calculateAdjustedTexCoord(vTexCoord, vec4(uClickPos, uClickPos + 0.1), aspectRatio);
  float numRipples = 1.75;
  float saturationFactor = 0.7;

  vec4 finalColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
  float totalMaskValue = 0.0f;
  vec4 edgeColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);

  // Process mask 0
  if(uNumMasks > 0) {
    float maskValue0 = texture(uMaskTexture0, tvTexCoord).r;
    vec4 saturatedColor = lowerSaturation(uMaskColor0 / 255.0, saturationFactor);
    vec4 plainColor = vec4(vec3(saturatedColor).rgb, 1.0);
    vec4 rippleColor = vec4(uMaskColor0.rgb / 255.0, 0.2);
    
    if (uActiveMask == 0 && uTime < timeThreshold) {
      float dist = length(adjustedClickCoord);
      float colorFactor = abs(sin((dist - uTime) * numRipples));
      plainColor = vec4(mix(rippleColor.rgb, plainColor.rgb, colorFactor), 1.0);
    }
    
    finalColor += maskValue0 * plainColor;
    totalMaskValue += maskValue0;

    edgeColor = detectEdges(uMaskTexture0, 1.25, uMaskColor0 / 255.0);
  }

  // Process mask 1
  if(uNumMasks > 1) {
    float maskValue1 = texture(uMaskTexture1, tvTexCoord).r;
    vec4 saturatedColor = lowerSaturation(uMaskColor1 / 255.0, saturationFactor);
    vec4 plainColor = vec4(vec3(saturatedColor).rgb, 1.0);
    vec4 rippleColor = vec4(uMaskColor1.rgb / 255.0, 0.2);
    
    if (uActiveMask == 1 && uTime < timeThreshold) {
      float dist = length(adjustedClickCoord);
      float colorFactor = abs(sin((dist - uTime) * numRipples));
      plainColor = vec4(mix(rippleColor.rgb, plainColor.rgb, colorFactor), 1.0);
    }
    
    finalColor += maskValue1 * plainColor;
    totalMaskValue += maskValue1;

    if(edgeColor.a <= 0.0f) {
      edgeColor = detectEdges(uMaskTexture1, 1.25, uMaskColor1 / 255.0);
    }
  }

  // Process mask 2
  if(uNumMasks > 2) {
    float maskValue2 = texture(uMaskTexture2, tvTexCoord).r;
    vec4 saturatedColor = lowerSaturation(uMaskColor2 / 255.0, saturationFactor);
    vec4 plainColor = vec4(vec3(saturatedColor).rgb, 1.0);
    vec4 rippleColor = vec4(uMaskColor2.rgb / 255.0, 0.2);
    
    if (uActiveMask == 2 && uTime < timeThreshold) {
      float dist = length(adjustedClickCoord);
      float colorFactor = abs(sin((dist - uTime) * numRipples));
      plainColor = vec4(mix(rippleColor.rgb, plainColor.rgb, colorFactor), 1.0);
    }
    
    finalColor += maskValue2 * plainColor;
    totalMaskValue += maskValue2;

    if(edgeColor.a <= 0.0f) {
      edgeColor = detectEdges(uMaskTexture2, 1.25, uMaskColor2 / 255.0);
    }
  }


  if (uNumMasks > 3) processMask(3, uMaskTexture3, uMaskColor3, finalColor, totalMaskValue, edgeColor);
  if (uNumMasks > 4) processMask(4, uMaskTexture4, uMaskColor4, finalColor, totalMaskValue, edgeColor);
  if (uNumMasks > 5) processMask(5, uMaskTexture5, uMaskColor5, finalColor, totalMaskValue, edgeColor);
  if (uNumMasks > 6) processMask(6, uMaskTexture6, uMaskColor6, finalColor, totalMaskValue, edgeColor);
  if (uNumMasks > 7) processMask(7, uMaskTexture7, uMaskColor7, finalColor, totalMaskValue, edgeColor);
  if (uNumMasks > 8) processMask(8, uMaskTexture8, uMaskColor8, finalColor, totalMaskValue, edgeColor);
  if (uNumMasks > 9) processMask(9, uMaskTexture9, uMaskColor9, finalColor, totalMaskValue, edgeColor);
  if (uNumMasks > 10) processMask(10, uMaskTexture10, uMaskColor10, finalColor, totalMaskValue, edgeColor);
  if (uNumMasks > 11) processMask(11, uMaskTexture11, uMaskColor11, finalColor, totalMaskValue, edgeColor);
  if (uNumMasks > 12) processMask(12, uMaskTexture12, uMaskColor12, finalColor, totalMaskValue, edgeColor);
  if (uNumMasks > 13) processMask(13, uMaskTexture13, uMaskColor13, finalColor, totalMaskValue, edgeColor);
  if (uNumMasks > 14) processMask(14, uMaskTexture14, uMaskColor14, finalColor, totalMaskValue, edgeColor);

  // Final color computation
  if(totalMaskValue > 0.0f) {
    finalColor.rgb /= totalMaskValue;
    finalColor.a = 1.0;
    finalColor = mix(color, finalColor, uOpacity);
  } else {
    finalColor = color;
  }

  if(edgeColor.a > 0.0f && uBorder) {
    finalColor = vec4(vec3(edgeColor), 1.0f);
  }
  
  fragColor = finalColor;
}