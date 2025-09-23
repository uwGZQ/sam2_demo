/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { useState } from 'react';
import { useAtomValue } from 'jotai';
import { objectAttributesAtom, objectCustomLabelsAtom, getObjectLabel } from '@/common/components/annotations/ObjectUtils';
import { BaseTracklet } from '@/common/tracker/Tracker';
import useVideo from '@/common/components/video/editor/useVideo';

// 定义下载状态类型
type DownloadingState = 'default' | 'downloading' | 'error';

type AttributeData = {
  name: string;
  value: string;
};

type MaskData = {
  frame: number;
  maskData: string; // Base64 encoded mask data
  boundaries: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
};

type TrackletExportData = {
  id: number;
  label: string;
  attributes: AttributeData[];
  masks: MaskData[];
};

export default function useMaskAttributesDownload() {
  const [downloadingState, setDownloadingState] = useState<DownloadingState>('default');
  const [progress, setProgress] = useState(0);
  const customLabels = useAtomValue(objectCustomLabelsAtom);
  const attributes = useAtomValue(objectAttributesAtom);
  const video = useVideo();

  const getMaskDataFromWorker = async (trackletId: number, frameIndex: number): Promise<{maskData: string, bounds?: {x: number, y: number, width: number, height: number}}> => {
    return new Promise((resolve) => {
      // Get reference to worker
      if (!video) {
        resolve({maskData: ''});
        return;
      }
      
      const worker = video.getWorker_ONLY_USE_WITH_CAUTION();
      
      // Set up one-time message handler to receive mask data
      const handleMessage = (event: MessageEvent) => {
        if (event.data.action === 'maskData' && 
            event.data.trackletId === trackletId && 
            event.data.frameIndex === frameIndex) {
          worker.removeEventListener('message', handleMessage);
          resolve({maskData: event.data.maskData});
        }
      };
      
      worker.addEventListener('message', handleMessage);
      
      // Request mask data from worker
      worker.postMessage({
        action: 'getMaskData',
        trackletId,
        frameIndex
      });
      
      // Set timeout to prevent hanging
      setTimeout(() => {
        worker.removeEventListener('message', handleMessage);
        resolve({maskData: ''});
      }, 1000);
    });
  };

  const downloadMasksAndAttributes = async () => {
    try {
      if (!video || !video.objects || video.objects.length === 0) {
        console.error('No objects available');
        setDownloadingState('error');
        return;
      }

      setDownloadingState('downloading');
      setProgress(0);

      const tracklets: BaseTracklet[] = video.objects; // 明确使用 BaseTracklet 类型
      const totalTracklets = tracklets.length;
      const exportData: TrackletExportData[] = [];
      
      for (let i = 0; i < totalTracklets; i++) {
        const tracklet: BaseTracklet = tracklets[i]; // 明确使用 BaseTracklet 类型
        const trackletId = tracklet.id;
        
        // Get label for tracklet
        const label = getObjectLabel(tracklet, customLabels);
        
        // Get attributes for tracklet
        const attributesArray: AttributeData[] = [];
        const trackletAttributes = attributes[trackletId] || [];
        
        for (const attr of trackletAttributes) {
          attributesArray.push({
            name: attr.name,
            value: attr.value
          });
        }
        
        // Get masks for tracklet
        const masks: MaskData[] = [];
        
        // We only have base tracklets without mask data in the main thread
        // Need to request actual mask data from the worker for each frame
        for (let frameIndex = 0; frameIndex < video.numberOfFrames; frameIndex++) {
          // Request actual mask data from worker for this frame
          const {maskData: maskDataBase64} = await getMaskDataFromWorker(trackletId, frameIndex);
          
          if (maskDataBase64) {
            // 尝试解析为 RLEObject
            let boundaries = {
              x: 0, 
              y: 0,
              width: video.width,
              height: video.height
            };
            
            try {
              // 如果 maskData 是 RLEObject 的 JSON 字符串，则尝试解析它获取更准确的边界
              const rleObject = JSON.parse(maskDataBase64);
              if (rleObject && rleObject.size && Array.isArray(rleObject.size)) {
                // 使用 RLEObject 的尺寸
                boundaries = {
                  x: 0,
                  y: 0,
                  width: rleObject.size[1],
                  height: rleObject.size[0]
                };
              }
            } catch (e) {
              // 如果不是有效的 JSON，保留默认边界
              console.log('Using default boundaries for mask');
            }
            
            masks.push({
              frame: frameIndex,
              maskData: maskDataBase64,
              boundaries
            });
          }
        }
        
        // Add tracklet data to export
        exportData.push({
          id: trackletId,
          label,
          attributes: attributesArray,
          masks
        });
        
        // Update progress
        setProgress((i + 1) / totalTracklets);
      }
      
      // Create and download JSON file
      const jsonContent = JSON.stringify(exportData, null, 2);
      const blob = new Blob([jsonContent], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `masks_and_attributes_${new Date().getTime()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      setDownloadingState('default');
      setProgress(0);
    } catch (error) {
      console.error('Error downloading masks and attributes:', error);
      setDownloadingState('error');
    }
  };

  return {
    downloadMasksAndAttributes,
    progress,
    state: downloadingState
  };
}