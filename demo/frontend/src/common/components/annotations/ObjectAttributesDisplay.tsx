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
import { useAtomValue } from 'jotai';
import { objectAttributesAtom } from '@/common/components/annotations/ObjectUtils';

type ObjectAttributesDisplayProps = {
  trackletId: number;
  className?: string;
  compact?: boolean;
};

export default function ObjectAttributesDisplay({
  trackletId,
  className = '',
  compact = false,
}: ObjectAttributesDisplayProps) {
  const attributes = useAtomValue(objectAttributesAtom);
  const objectAttributes = attributes[trackletId] || [];

  if (objectAttributes.length === 0) {
    return null;
  }

  if (compact) {
    // Compact display for tight spaces (like tooltips)
    return (
      <div className={`text-xs ${className}`}>
        {objectAttributes.map((attr, index) => (
          <span key={index} className="mr-2">
            {attr.name}: <span className="font-medium">{attr.value}</span>
            {index < objectAttributes.length - 1 ? ',' : ''}
          </span>
        ))}
      </div>
    );
  }

  // Full display with better formatting
  return (
    <div className={`space-y-1 ${className}`}>
      {objectAttributes.map((attr, index) => (
        <div key={index} className="text-sm">
          <span className="font-medium">{attr.name}:</span>{' '}
          <span>{attr.value}</span>
        </div>
      ))}
    </div>
  );
}