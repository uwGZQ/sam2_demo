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
import {BaseTracklet} from '@/common/tracker/Tracker';
import {atom} from 'jotai';
import {AttributeItem} from '@/common/components/annotations/ObjectAttributesEditor';

// Create an atom to store custom labels for objects
export const objectCustomLabelsAtom = atom<Record<number, string>>({});

// Export the attributes type for use in other components
export type {AttributeItem} from '@/common/components/annotations/ObjectAttributesEditor';

// Export predefined attributes
export {PREDEFINED_ATTRIBUTES} from '@/common/components/annotations/ObjectAttributesEditor';

// Create an atom to store attributes for objects (re-exported for consistency)
export {objectAttributesAtom} from '@/common/components/annotations/ObjectAttributesEditor';

export function getObjectLabel(tracklet: BaseTracklet, customLabels?: Record<number, string>) {
  // If a custom label exists for this tracklet, use it
  if (customLabels && customLabels[tracklet.id] !== undefined) {
    return customLabels[tracklet.id];
  }
  // Otherwise use the default label
  return `Object ${tracklet.id + 1}`;
}

// Helper function to get all attributes for an object
export function getObjectAttributes(trackletId: number, attributes: Record<number, AttributeItem[]>): AttributeItem[] {
  return attributes[trackletId] || [];
}

// Helper function to get a specific attribute value by name
export function getObjectAttributeValue(
  trackletId: number, 
  attributeName: string, 
  attributes: Record<number, AttributeItem[]>
): string | null {
  const objectAttrs = attributes[trackletId] || [];
  const attr = objectAttrs.find(a => a.name.toLowerCase() === attributeName.toLowerCase());
  return attr ? attr.value : null;
}