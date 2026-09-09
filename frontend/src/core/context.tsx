import { FC, PropsWithChildren, createContext, useCallback, useEffect, useState } from 'react';

import {
  ActiveModel,
  DisplayConfig,
  ElementHistoryPoint,
  GenerateConfig,
  NotificationType,
  ProjectStateModel,
  ProjectionOutModel,
  SelectionConfig,
} from '../types';
import { DEFAULT_CONTEXT } from './useAppContext';

// Context content
export type AppContextValue = {
  notifications: NotificationType[]; // manage notification
  selectionConfig: SelectionConfig; // selection for the next element
  generateConfig: GenerateConfig;
  displayConfig: DisplayConfig; // config for the visual
  currentProject?: ProjectStateModel | null; // current project selected
  currentScheme?: string; // scheme selected to annotate
  currentProjection?: ProjectionOutModel;
  currentProjectionName?: string | null;
  labelColorMapping?: { [key: string]: string };
  activeModel?: ActiveModel | null;
  freqRefreshQuickModel: number; // freq to refresh active learning model
  history: ElementHistoryPoint[]; // element annotated
  selectionHistory: Record<string, string>; // history of the selection
  reFetchCurrentProject?: () => void; // update the state of the project
  phase: string;
  isComputing: boolean;
  developmentMode: boolean;
};

export const CONTEXT_LOCAL_STORAGE_KEY = 'activeTigger.context';
// developmentMode lives in its own key so it survives logout (it's a per-user-on-this-machine
// preference, not a session value that should be wiped with the rest of the context).
export const DEVELOPMENT_MODE_STORAGE_KEY = 'activeTigger.developmentMode';

const storedContext = localStorage.getItem(CONTEXT_LOCAL_STORAGE_KEY);

const readStoredDevelopmentMode = (): boolean =>
  localStorage.getItem(DEVELOPMENT_MODE_STORAGE_KEY) === 'true';

// type of the context
export type AppContextType = {
  appContext: AppContextValue;
  setAppContext: React.Dispatch<React.SetStateAction<AppContextValue>>;
  resetContext: () => void;
};

export const AppContext = createContext<AppContextType>(null as unknown as AppContextType);

const _useAppContext = () => {
  const [appContext, setAppContext] = useState<AppContextValue>(() => {
    const base = storedContext ? (JSON.parse(storedContext) as AppContextValue) : DEFAULT_CONTEXT;
    return { ...base, developmentMode: readStoredDevelopmentMode() };
  });

  //store context in localstorage
  useEffect(() => {
    // currentProjection holds one node per element and can exceed the localStorage
    // quota on large datasets; it is refetched from the API so it is not persisted.
    const persistedContext = { ...appContext, currentProjection: undefined };
    try {
      localStorage.setItem(DEVELOPMENT_MODE_STORAGE_KEY, String(appContext.developmentMode));
      localStorage.setItem(CONTEXT_LOCAL_STORAGE_KEY, JSON.stringify(persistedContext));
    } catch (e) {
      console.warn('Could not persist app context to localStorage', e);
    }
  }, [appContext]);

  // Function to reset the context
  const resetContext = useCallback(() => {
    const newContext = DEFAULT_CONTEXT;
    // keep the interface type
    newContext.displayConfig.interfaceType = appContext.displayConfig.interfaceType;
    // keep the experimental (developmentMode) toggle across navigation
    newContext.developmentMode = appContext.developmentMode;
    setAppContext(newContext);
  }, [appContext.displayConfig.interfaceType, appContext.developmentMode, setAppContext]);

  return {
    appContext,
    setAppContext,
    resetContext,
  };
};

export const AppContextProvider: FC<PropsWithChildren> = ({ children }) => {
  const context = _useAppContext();

  return <AppContext.Provider value={context}>{children}</AppContext.Provider>;
};
