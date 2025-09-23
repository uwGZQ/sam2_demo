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
import { useState, useRef, useEffect } from 'react';
import { useAtom } from 'jotai';
import { Add, Edit, TrashCan } from '@carbon/icons-react';
import { atom } from 'jotai';

// Define the structure for object attributes
export type AttributeItem = {
  name: string;
  value: string;
};

// Create an atom to store attributes for objects
export const objectAttributesAtom = atom<Record<number, AttributeItem[]>>({});

// Predefined attribute names
export const PREDEFINED_ATTRIBUTES = [
  'color',
  'shape',
  'size',
  'texture',
  'state',
  'material',
  'pattern',
  'position',
  'orientation',
  'motion'
];

type ObjectAttributesEditorProps = {
  trackletId: number;
};

export default function ObjectAttributesEditor({ trackletId }: ObjectAttributesEditorProps) {
  const [attributes, setAttributes] = useAtom(objectAttributesAtom);
  const [isAddingAttribute, setIsAddingAttribute] = useState(false);
  const [newAttributeName, setNewAttributeName] = useState('');
  const [newAttributeValue, setNewAttributeValue] = useState('');
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [isCustomAttributeName, setIsCustomAttributeName] = useState(false);
  const inputNameRef = useRef<HTMLInputElement>(null);
  const inputValueRef = useRef<HTMLInputElement>(null);
  const selectRef = useRef<HTMLSelectElement>(null);

  const objectAttributes = attributes[trackletId] || [];

  // Get list of attribute names that haven't been used yet
  const unusedPredefinedAttributes = PREDEFINED_ATTRIBUTES.filter(
    name => !objectAttributes.some(attr => attr.name.toLowerCase() === name.toLowerCase())
  );

  // Focus on appropriate input when adding a new attribute
  useEffect(() => {
    if (isAddingAttribute) {
      if (isCustomAttributeName && inputNameRef.current) {
        inputNameRef.current.focus();
      } else if (selectRef.current && unusedPredefinedAttributes.length > 0) {
        selectRef.current.focus();
      } else if (inputValueRef.current) {
        inputValueRef.current.focus();
      }
    }
  }, [isAddingAttribute, isCustomAttributeName, unusedPredefinedAttributes.length]);

  // Focus on input when editing an attribute
  useEffect(() => {
    if (editingIndex !== null && inputValueRef.current) {
      inputValueRef.current.focus();
    }
  }, [editingIndex]);

  const handleAddAttribute = () => {
    setIsAddingAttribute(true);
    setNewAttributeName(unusedPredefinedAttributes.length > 0 ? unusedPredefinedAttributes[0] : '');
    setNewAttributeValue('');
    setIsCustomAttributeName(unusedPredefinedAttributes.length === 0);
  };

  const handleSaveAttribute = () => {
    const trimmedName = newAttributeName.trim();
    const trimmedValue = newAttributeValue.trim();
    
    if (trimmedName && trimmedValue) {
      const updatedAttributes = [...(attributes[trackletId] || [])];
      
      if (editingIndex !== null) {
        // Update existing attribute
        updatedAttributes[editingIndex] = { name: trimmedName, value: trimmedValue };
      } else {
        // Add new attribute
        updatedAttributes.push({ name: trimmedName, value: trimmedValue });
      }
      
      setAttributes({
        ...attributes,
        [trackletId]: updatedAttributes,
      });
    }
    
    setIsAddingAttribute(false);
    setEditingIndex(null);
    setIsCustomAttributeName(false);
  };

  const handleEditAttribute = (index: number) => {
    const attribute = objectAttributes[index];
    setNewAttributeName(attribute.name);
    setNewAttributeValue(attribute.value);
    setEditingIndex(index);
    setIsAddingAttribute(true);
    // For editing, we always set to true since we're editing an existing name
    setIsCustomAttributeName(true);
  };

  const handleDeleteAttribute = (index: number) => {
    const updatedAttributes = [...objectAttributes];
    updatedAttributes.splice(index, 1);
    
    setAttributes({
      ...attributes,
      [trackletId]: updatedAttributes,
    });
  };

  const handleCancel = () => {
    setIsAddingAttribute(false);
    setEditingIndex(null);
    setIsCustomAttributeName(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    e.stopPropagation(); // Prevent container from handling keyboard events
    if (e.key === 'Enter') {
      handleSaveAttribute();
    } else if (e.key === 'Escape') {
      handleCancel();
    }
  };

  const handleToggleCustomAttribute = () => {
    setIsCustomAttributeName(!isCustomAttributeName);
    setNewAttributeName('');
  };

  const handleAttributeNameChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    if (value === 'custom') {
      setIsCustomAttributeName(true);
      setNewAttributeName('');
    } else {
      setNewAttributeName(value);
    }
  };

  return (
    <div className="mt-2 ml-2">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-400">Attributes</span>
        {!isAddingAttribute && (
          <button
            onClick={handleAddAttribute}
            className="p-1 hover:bg-gray-700 rounded-full">
            <Add size={16} />
          </button>
        )}
      </div>

      {/* Attribute form when adding/editing */}
      {isAddingAttribute && (
        <div className="mb-2 bg-gray-800 rounded p-2">
          {/* Attribute name input (either select or custom input) */}
          <div className="mb-1">
            {!isCustomAttributeName && editingIndex === null && unusedPredefinedAttributes.length > 0 ? (
              <div className="flex mb-1">
                <select
                  ref={selectRef}
                  className="bg-gray-700 text-white px-2 py-1 rounded flex-grow"
                  value={newAttributeName}
                  onChange={handleAttributeNameChange}
                  onKeyDown={handleKeyDown}
                >
                  {unusedPredefinedAttributes.map(attr => (
                    <option key={attr} value={attr}>
                      {attr.charAt(0).toUpperCase() + attr.slice(1)}
                    </option>
                  ))}
                  <option value="custom">Custom attribute...</option>
                </select>
              </div>
            ) : (
              <div className="mb-1">
                {editingIndex === null && (
                  <div className="flex items-center mb-1">
                    <input
                      ref={inputNameRef}
                      className="bg-gray-700 text-white px-2 py-1 rounded w-full"
                      placeholder="Attribute name"
                      value={newAttributeName}
                      onChange={(e) => setNewAttributeName(e.target.value)}
                      onKeyDown={handleKeyDown}
                      maxLength={20}
                    />
                    {unusedPredefinedAttributes.length > 0 && (
                      <button
                        onClick={handleToggleCustomAttribute}
                        className="ml-2 text-xs text-blue-400 hover:text-blue-300"
                      >
                        Use preset
                      </button>
                    )}
                  </div>
                )}
              </div>
            )}
            
            {/* Value input is always shown */}
            <input
              ref={inputValueRef}
              className="bg-gray-700 text-white px-2 py-1 rounded w-full"
              placeholder="Value"
              value={newAttributeValue}
              onChange={(e) => setNewAttributeValue(e.target.value)}
              onKeyDown={handleKeyDown}
              maxLength={30}
            />
          </div>
          
          {/* Action buttons */}
          <div className="flex justify-end gap-2">
            <button
              onClick={handleCancel}
              className="text-xs text-gray-400 hover:text-white">
              Cancel
            </button>
            <button
              onClick={handleSaveAttribute}
              className="text-xs text-blue-400 hover:text-blue-300">
              Save
            </button>
          </div>
        </div>
      )}

      {/* List of existing attributes */}
      {objectAttributes.length > 0 && (
        <div className="space-y-1">
          {objectAttributes.map((attr, index) => (
            <div
              key={index}
              className="flex items-center justify-between bg-gray-800 rounded px-2 py-1 text-sm">
              <div className="flex-1 overflow-hidden">
                <span className="font-medium">{attr.name}:</span>{' '}
                <span className="text-gray-300">{attr.value}</span>
              </div>
              <div className="flex gap-1">
                <button
                  onClick={() => handleEditAttribute(index)}
                  className="p-1 hover:bg-gray-700 rounded-full">
                  <Edit size={14} />
                </button>
                <button
                  onClick={() => handleDeleteAttribute(index)}
                  className="p-1 hover:bg-gray-700 rounded-full">
                  <TrashCan size={14} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Empty state when no attributes */}
      {!isAddingAttribute && objectAttributes.length === 0 && (
        <div className="text-sm text-gray-500">
          No attributes. Click + to add.
        </div>
      )}
    </div>
  );
}