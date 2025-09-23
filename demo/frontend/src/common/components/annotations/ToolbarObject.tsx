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
import ObjectActions from '@/common/components/annotations/ObjectActions';
import ObjectPlaceholder from '@/common/components/annotations/ObjectPlaceholder';
import ObjectThumbnail from '@/common/components/annotations/ObjectThumbnail';
import ToolbarObjectContainer from '@/common/components/annotations/ToolbarObjectContainer';
import useVideo from '@/common/components/video/editor/useVideo';
import ObjectLabelEditor from '@/common/components/annotations/ObjectLabelEditor';
import ObjectAttributesEditor from '@/common/components/annotations/ObjectAttributesEditor';
import {BaseTracklet} from '@/common/tracker/Tracker';
import emptyFunction from '@/common/utils/emptyFunction';
import {getObjectLabel, objectCustomLabelsAtom} from '@/common/components/annotations/ObjectUtils';
import {activeTrackletObjectIdAtom} from '@/demo/atoms';
import {useSetAtom, useAtomValue} from 'jotai';
import {ReactNode, useState} from 'react';
import {ChevronDown, ChevronUp} from '@carbon/icons-react';

type Props = {
  label: string;
  tracklet: BaseTracklet;
  isActive: boolean;
  isMobile?: boolean;
  onClick?: () => void;
  onThumbnailClick?: () => void;
};

export default function ToolbarObject({
  label,
  tracklet,
  isActive,
  isMobile = false,
  onClick,
  onThumbnailClick = emptyFunction,
}: Props) {
  const video = useVideo();
  const setActiveTrackletId = useSetAtom(activeTrackletObjectIdAtom);
  const customLabels = useAtomValue(objectCustomLabelsAtom);
  const [showAttributes, setShowAttributes] = useState(false);
  
  async function handleCancelNewObject() {
    try {
      await video?.deleteTracklet(tracklet.id);
    } catch (error) {
      reportError(error);
    } finally {
      setActiveTrackletId(null);
    }
  }

  const toggleAttributes = (e: React.MouseEvent) => {
    e.stopPropagation();
    setShowAttributes(!showAttributes);
  };

  if (!tracklet.isInitialized) {
    return (
      <ToolbarObjectContainer
        alignItems="center"
        isActive={isActive}
        title="New object"
        subtitle="No object is currently selected. Click an object in the video."
        thumbnail={<ObjectPlaceholder showPlus={false} />}
        isMobile={isMobile}
        onClick={onClick}
        onCancel={handleCancelNewObject}
      />
    );
  }
  
  let titleComponent: ReactNode = null;
  if (isMobile) {
    titleComponent = getObjectLabel(tracklet, customLabels);
  } else {
    titleComponent = <ObjectLabelEditor 
      trackletId={tracklet.id} 
      defaultLabel={label} 
    />;
  }
  
  return (
    <ToolbarObjectContainer
      isActive={isActive}
      onClick={onClick}
      title={titleComponent}
      subtitle=""
      thumbnail={
        <ObjectThumbnail
          thumbnail={tracklet.thumbnail}
          color={tracklet.color}
          onClick={onThumbnailClick}
        />
      }
      isMobile={isMobile}>
      <div className="w-full">
        <ObjectActions objectId={tracklet.id} active={isActive} />
        
        {isActive && !isMobile && (
          <div className="mt-3">
            <div 
              className="flex items-center justify-between cursor-pointer hover:bg-gray-800 rounded px-2 py-1"
              onClick={toggleAttributes}
            >
              <span className="text-sm font-medium">
                {showAttributes ? "Hide attributes" : "Show attributes"}
              </span>
              {showAttributes ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </div>
            
            {showAttributes && (
              <ObjectAttributesEditor trackletId={tracklet.id} />
            )}
          </div>
        )}
      </div>
    </ToolbarObjectContainer>
  );
}