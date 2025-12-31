/**
 * User Role Context Provider and Hook
 *
 * Manages user role state with localStorage persistence.
 * Used for frontend role-based access control to hide admin-only features.
 */

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  ReactNode,
} from 'react';

// Available roles in the system
export type UserRole = 'viewer' | 'analyst' | 'admin';

const STORAGE_KEY = 'userRole';
const DEFAULT_ROLE: UserRole = 'viewer';

interface UserRoleContextValue {
  userRole: UserRole;
  setUserRole: (role: UserRole) => void;
  isAdmin: boolean;
  isAnalyst: boolean;
  isViewer: boolean;
  canEdit: boolean; // analyst or admin
  canManage: boolean; // admin only
}

const UserRoleContext = createContext<UserRoleContextValue | undefined>(
  undefined
);

interface UserRoleProviderProps {
  children: ReactNode;
}

export function UserRoleProvider({ children }: UserRoleProviderProps) {
  const [userRole, setUserRoleState] = useState<UserRole>(() => {
    // Initialize from localStorage or environment variable
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored && ['viewer', 'analyst', 'admin'].includes(stored)) {
        return stored as UserRole;
      }
    }
    // Fall back to environment variable
    const envRole = import.meta.env.VITE_USER_ROLE;
    if (envRole && ['viewer', 'analyst', 'admin'].includes(envRole)) {
      return envRole as UserRole;
    }
    return DEFAULT_ROLE;
  });

  // Persist to localStorage when state changes
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, userRole);
  }, [userRole]);

  const setUserRole = useCallback((role: UserRole) => {
    setUserRoleState(role);
  }, []);

  // Role-based permission helpers
  const isAdmin = userRole === 'admin';
  const isAnalyst = userRole === 'analyst';
  const isViewer = userRole === 'viewer';
  const canEdit = isAdmin || isAnalyst;
  const canManage = isAdmin;

  return (
    <UserRoleContext.Provider
      value={{
        userRole,
        setUserRole,
        isAdmin,
        isAnalyst,
        isViewer,
        canEdit,
        canManage,
      }}
    >
      {children}
    </UserRoleContext.Provider>
  );
}

export function useUserRole(): UserRoleContextValue {
  const context = useContext(UserRoleContext);
  if (context === undefined) {
    throw new Error('useUserRole must be used within a UserRoleProvider');
  }
  return context;
}

/**
 * Helper function to get user role from localStorage.
 * Used by components that may not have access to React context.
 */
export function getUserRoleFromStorage(): UserRole {
  if (typeof window !== 'undefined') {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored && ['viewer', 'analyst', 'admin'].includes(stored)) {
      return stored as UserRole;
    }
  }
  const envRole = import.meta.env.VITE_USER_ROLE;
  if (envRole && ['viewer', 'analyst', 'admin'].includes(envRole)) {
    return envRole as UserRole;
  }
  return DEFAULT_ROLE;
}

/**
 * Check if current user can perform edit operations (analyst or admin).
 */
export function canUserEdit(): boolean {
  const role = getUserRoleFromStorage();
  return role === 'admin' || role === 'analyst';
}

/**
 * Check if current user can perform management operations (admin only).
 */
export function canUserManage(): boolean {
  const role = getUserRoleFromStorage();
  return role === 'admin';
}
