/**
 * Environment Configuration Types
 *
 * Type definitions for the setup wizard that helps users
 * configure the env.template file.
 */

// Application Environment
export type AppEnvironment = 'local' | 'development' | 'staging' | 'production';
export type UserRole = 'admin' | 'operator' | 'viewer';

// Complete environment configuration
export interface EnvironmentConfig {
  // Application Environment
  app_env: AppEnvironment;

  // API Security & Tenant Context
  require_api_key: boolean;
  api_key: string;
  api_key_role: UserRole;
  trust_client_headers: boolean;
  require_tenant_header: boolean;
  default_tenant_id: string;
  default_user_role: UserRole;
  tenant_id_allowlist: string;
  allow_in_process_training: boolean;

  // Observability
  enable_otel: boolean;
  otel_exporter_otlp_endpoint: string;
  otel_exporter_otlp_insecure: boolean;
  otel_service_name: string;
  enable_metrics: boolean;
  enable_db_metrics: boolean;
  enable_tenant_metrics: boolean;
  tenant_metrics_allowlist: string;
  results_db_use_migrations: boolean;

  // Frontend Configuration
  frontend_port: number;
  frontend_dev_port: number;
  vite_tenant_id: string;
  vite_api_key: string;
  vite_user_id: string;
  vite_user_role: UserRole;

  // PostgreSQL Backend Database
  backend_db_host: string;
  backend_db_port: number;
  backend_db_name: string;
  backend_db_user: string;
  backend_db_pass: string;
  backend_db_connect_timeout: number;
  backend_db_statement_timeout_ms: number;

  // SOTI XSight SQL Server Configuration
  dw_db_host: string;
  dw_db_port: number;
  dw_db_name: string;
  dw_db_user: string;
  dw_db_pass: string;
  dw_db_driver: string;
  dw_db_connect_timeout: number;
  dw_db_query_timeout: number;
  dw_trust_server_cert: boolean;

  // SOTI MobiControl SQL Server Configuration
  mc_db_host: string;
  mc_db_port: number;
  mc_db_name: string;
  mc_db_user: string;
  mc_db_pass: string;
  mc_db_driver: string;
  mc_db_connect_timeout: number;
  mc_db_query_timeout: number;
  mc_trust_server_cert: boolean;

  // LLM Configuration
  enable_llm: boolean;
  llm_base_url: string;
  llm_api_key: string;
  llm_model_name: string;
  llm_api_version: string;
  llm_base_url_allowlist: string;

  // Real-Time Streaming Configuration
  enable_streaming: boolean;
  redis_url: string;
  redis_db: number;
  stream_buffer_size: number;
  stream_flush_interval_ms: number;

  // SOTI MobiControl API Configuration
  mobicontrol_server_url: string;
  mobicontrol_client_id: string;
  mobicontrol_client_secret: string;
  mobicontrol_username: string;
  mobicontrol_password: string;
  mobicontrol_tenant_id: string;
}

// Wizard step configuration
export interface SetupStep {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  fields: (keyof EnvironmentConfig)[];
}

// Field metadata for rendering forms
export interface FieldConfig {
  key: keyof EnvironmentConfig;
  label: string;
  type: 'text' | 'password' | 'number' | 'boolean' | 'select';
  placeholder?: string;
  description?: string;
  required?: boolean;
  options?: { value: string; label: string }[];
  defaultValue?: string | number | boolean;
  validation?: {
    min?: number;
    max?: number;
    pattern?: string;
    message?: string;
  };
}

// API response types
export interface SaveConfigResponse {
  success: boolean;
  message: string;
  file_path?: string;
}

export interface ValidateConfigResponse {
  valid: boolean;
  errors: { field: string; message: string }[];
  warnings: { field: string; message: string }[];
}

export interface TestConnectionRequest {
  type: 'backend_db' | 'dw_db' | 'mc_db' | 'llm' | 'redis' | 'mobicontrol_api';
  config: Partial<EnvironmentConfig>;
}

export interface TestConnectionResponse {
  success: boolean;
  message: string;
  latency_ms?: number;
}

// Default configuration values
export const defaultEnvironmentConfig: EnvironmentConfig = {
  // Application Environment
  app_env: 'local',

  // API Security & Tenant Context
  require_api_key: false,
  api_key: '',
  api_key_role: 'viewer',
  trust_client_headers: false,
  require_tenant_header: false,
  default_tenant_id: 'default',
  default_user_role: 'viewer',
  tenant_id_allowlist: '',
  allow_in_process_training: false,

  // Observability
  enable_otel: false,
  otel_exporter_otlp_endpoint: '',
  otel_exporter_otlp_insecure: true,
  otel_service_name: 'stella-sentinel-api',
  enable_metrics: true,
  enable_db_metrics: true,
  enable_tenant_metrics: false,
  tenant_metrics_allowlist: '',
  results_db_use_migrations: false,

  // Frontend Configuration
  frontend_port: 3000,
  frontend_dev_port: 5173,
  vite_tenant_id: 'default',
  vite_api_key: '',
  vite_user_id: '',
  vite_user_role: 'viewer',

  // PostgreSQL Backend Database
  backend_db_host: 'postgres',
  backend_db_port: 5432,
  backend_db_name: 'anomaly_detection',
  backend_db_user: 'postgres',
  backend_db_pass: 'postgres',
  backend_db_connect_timeout: 5,
  backend_db_statement_timeout_ms: 30000,

  // SOTI XSight SQL Server Configuration
  dw_db_host: '',
  dw_db_port: 1433,
  dw_db_name: 'XSight',
  dw_db_user: '',
  dw_db_pass: '',
  dw_db_driver: 'ODBC Driver 18 for SQL Server',
  dw_db_connect_timeout: 5,
  dw_db_query_timeout: 30,
  dw_trust_server_cert: false,

  // SOTI MobiControl SQL Server Configuration
  mc_db_host: '',
  mc_db_port: 1433,
  mc_db_name: 'MobiControlDB',
  mc_db_user: '',
  mc_db_pass: '',
  mc_db_driver: 'ODBC Driver 18 for SQL Server',
  mc_db_connect_timeout: 5,
  mc_db_query_timeout: 30,
  mc_trust_server_cert: false,

  // LLM Configuration
  enable_llm: true,
  llm_base_url: 'http://ollama:11434',
  llm_api_key: 'not-needed',
  llm_model_name: 'llama3.2',
  llm_api_version: '',
  llm_base_url_allowlist: '',

  // Real-Time Streaming Configuration
  enable_streaming: false,
  redis_url: 'redis://redis:6379',
  redis_db: 0,
  stream_buffer_size: 1000,
  stream_flush_interval_ms: 100,

  // SOTI MobiControl API Configuration
  mobicontrol_server_url: '',
  mobicontrol_client_id: '',
  mobicontrol_client_secret: '',
  mobicontrol_username: '',
  mobicontrol_password: '',
  mobicontrol_tenant_id: '',
};
