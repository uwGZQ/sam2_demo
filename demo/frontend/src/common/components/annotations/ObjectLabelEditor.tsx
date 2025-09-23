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
import {useState, useRef, useEffect} from 'react';
// 使用相对导入路径来避免路径问题
import {objectCustomLabelsAtom} from '@/common/components/annotations/ObjectUtils';
import {useAtom} from 'jotai';
import {Edit} from '@carbon/icons-react';

type ObjectLabelEditorProps = {
  trackletId: number;
  defaultLabel: string;
};

export default function ObjectLabelEditor({trackletId, defaultLabel}: ObjectLabelEditorProps) {
  const [customLabels, setCustomLabels] = useAtom(objectCustomLabelsAtom);
  const [isEditing, setIsEditing] = useState(false);
  const [inputValue, setInputValue] = useState(customLabels[trackletId] || defaultLabel);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isEditing]);

  const handleEditClick = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent container click from triggering
    setIsEditing(true);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const saveLabel = () => {
    const trimmedValue = inputValue.trim();
    if (trimmedValue && trimmedValue !== defaultLabel) {
      // Save to atom if it's different from default
      setCustomLabels({...customLabels, [trackletId]: trimmedValue});
    } else if (customLabels[trackletId]) {
      // Remove from custom labels if it's the same as default
      const newLabels = {...customLabels};
      delete newLabels[trackletId];
      setCustomLabels(newLabels);
    }
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    e.stopPropagation(); // Prevent container from handling keyboard events
    if (e.key === 'Enter') {
      saveLabel();
    } else if (e.key === 'Escape') {
      setIsEditing(false);
      setInputValue(customLabels[trackletId] || defaultLabel);
    }
  };

  const handleBlur = () => {
    saveLabel();
  };

  const handleInputClick = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent container click from triggering
  };

  if (isEditing) {
    return (
      <input
        ref={inputRef}
        className="bg-gray-800 text-white px-2 py-1 rounded w-full"
        value={inputValue}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        onBlur={handleBlur}
        onClick={handleInputClick}
        maxLength={30}
      />
    );
  }

  return (
    <div className="flex items-center gap-2">
      <span>{customLabels[trackletId] || defaultLabel}</span>
      <button
        onClick={handleEditClick}
        className="p-1 hover:bg-gray-700 rounded-full">
        <Edit size={16} />
      </button>
    </div>
  );
}