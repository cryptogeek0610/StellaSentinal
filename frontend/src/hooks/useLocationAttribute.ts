import { useState, useEffect } from 'react';

const LOCATION_ATTRIBUTE_KEY = 'locationAttributeName';

export function useLocationAttribute() {
  const [attributeName, setAttributeName] = useState<string>(() => {
    const saved = localStorage.getItem(LOCATION_ATTRIBUTE_KEY);
    return saved || 'Store';
  });

  useEffect(() => {
    localStorage.setItem(LOCATION_ATTRIBUTE_KEY, attributeName);
  }, [attributeName]);

  return [attributeName, setAttributeName] as const;
}

