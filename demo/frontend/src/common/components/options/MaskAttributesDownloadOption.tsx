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

import OptionButton from '@/common/components/options/OptionButton';
import useMaskAttributesDownload from './useMaskAttributesDownload';
import { Download } from '@carbon/icons-react';

export default function MaskAttributesDownloadOption() {
  const { downloadMasksAndAttributes, state, progress } = useMaskAttributesDownload();

  const isDisabled = state === 'downloading' || state === 'error';
  const title = state === 'downloading' 
    ? `Downloading Masks (${Math.round(progress * 100)}%)` 
    : state === 'error' 
      ? 'Download Failed' 
      : 'Download Masks & Attributes';

  const handleClick = () => {
    if (!isDisabled) {
      downloadMasksAndAttributes();
    }
  };

  return (
    <OptionButton
      title={title}
      Icon={Download}
      variant="default"
      isDisabled={isDisabled}
      onClick={handleClick}
    />
  );
} 