-- ============================================================================
-- SOTI Anomaly Detection - Backend Database Schema
-- ============================================================================
-- This script creates the backend database schema for the anomaly detection
-- application. It supports multi-tenant data isolation, multiple data sources
-- (XSight, MobiControl), and LLM-powered explanations.
--
-- Usage:
--   docker-compose exec sqlserver /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P "YourPassword" -i /app/scripts/init_backend_schema.sql
-- ============================================================================

-- Create database if it doesn't exist
IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = 'SOTI_AnomalyDetection')
BEGIN
    CREATE DATABASE SOTI_AnomalyDetection;
    PRINT 'Database SOTI_AnomalyDetection created.';
END
ELSE
BEGIN
    PRINT 'Database SOTI_AnomalyDetection already exists.';
END
GO

USE SOTI_AnomalyDetection;
GO

PRINT 'Creating backend database schema...';
GO

-- ============================================================================
-- Drop existing tables (in reverse dependency order) for clean recreation
-- WARNING: This will delete all data! Comment out for production.
-- ============================================================================
IF OBJECT_ID('alert_notifications', 'U') IS NOT NULL DROP TABLE alert_notifications;
IF OBJECT_ID('alerts', 'U') IS NOT NULL DROP TABLE alerts;
IF OBJECT_ID('alert_rules', 'U') IS NOT NULL DROP TABLE alert_rules;
IF OBJECT_ID('cost_calculation_cache', 'U') IS NOT NULL DROP TABLE cost_calculation_cache;
IF OBJECT_ID('cost_audit_logs', 'U') IS NOT NULL DROP TABLE cost_audit_logs;
IF OBJECT_ID('operational_costs', 'U') IS NOT NULL DROP TABLE operational_costs;
IF OBJECT_ID('device_type_costs', 'U') IS NOT NULL DROP TABLE device_type_costs;
IF OBJECT_ID('model_metrics', 'U') IS NOT NULL DROP TABLE model_metrics;
IF OBJECT_ID('model_deployments', 'U') IS NOT NULL DROP TABLE model_deployments;
IF OBJECT_ID('ml_models', 'U') IS NOT NULL DROP TABLE ml_models;
IF OBJECT_ID('device_patterns', 'U') IS NOT NULL DROP TABLE device_patterns;
IF OBJECT_ID('anomaly_events', 'U') IS NOT NULL DROP TABLE anomaly_events;
IF OBJECT_ID('audit_logs', 'U') IS NOT NULL DROP TABLE audit_logs;
IF OBJECT_ID('explanations', 'U') IS NOT NULL DROP TABLE explanations;
IF OBJECT_ID('change_log', 'U') IS NOT NULL DROP TABLE change_log;
IF OBJECT_ID('anomalies', 'U') IS NOT NULL DROP TABLE anomalies;
IF OBJECT_ID('baselines', 'U') IS NOT NULL DROP TABLE baselines;
IF OBJECT_ID('telemetry_points', 'U') IS NOT NULL DROP TABLE telemetry_points;
IF OBJECT_ID('metric_definitions', 'U') IS NOT NULL DROP TABLE metric_definitions;
IF OBJECT_ID('devices', 'U') IS NOT NULL DROP TABLE devices;
IF OBJECT_ID('users', 'U') IS NOT NULL DROP TABLE users;
IF OBJECT_ID('tenants', 'U') IS NOT NULL DROP TABLE tenants;
GO

-- ============================================================================
-- Core Tables
-- ============================================================================

-- Tenants (multi-tenant isolation)
CREATE TABLE tenants (
    tenant_id VARCHAR(50) PRIMARY KEY,
    name NVARCHAR(255) NOT NULL,
    tier VARCHAR(20) DEFAULT 'standard',  -- free, standard, enterprise
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    metadata NVARCHAR(MAX)  -- JSON
);
PRINT 'Table tenants created.';
GO

-- Users (authentication and RBAC)
CREATE TABLE users (
    user_id VARCHAR(50) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    name NVARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL,  -- viewer, analyst, admin
    password_hash VARCHAR(255),
    is_active BIT DEFAULT 1,
    last_login DATETIME2,
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    metadata NVARCHAR(MAX),  -- JSON: preferences, settings
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id)
);
CREATE INDEX idx_tenant ON users(tenant_id);
CREATE INDEX idx_email ON users(email);
CREATE INDEX idx_role ON users(role);
PRINT 'Table users created.';
GO

-- Devices (multi-source: XSight, MobiControl, etc.)
CREATE TABLE devices (
    device_id VARCHAR(50) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    source VARCHAR(20) NOT NULL,  -- xsight, mobicontrol, synthetic
    external_id VARCHAR(100) NOT NULL,
    name NVARCHAR(255),
    device_type VARCHAR(50),
    os_version VARCHAR(50),
    last_seen DATETIME2,
    device_group_id VARCHAR(50),  -- For future grouping support
    metadata NVARCHAR(MAX),  -- JSON
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id)
);
CREATE INDEX idx_tenant_source ON devices(tenant_id, source);
CREATE INDEX idx_external ON devices(external_id, source);
CREATE INDEX idx_last_seen ON devices(last_seen);
CREATE INDEX idx_device_group ON devices(device_group_id);
PRINT 'Table devices created.';
GO

-- Metric Definitions (standard + custom datapoints)
CREATE TABLE metric_definitions (
    metric_id VARCHAR(50) PRIMARY KEY,
    name NVARCHAR(100) NOT NULL,
    category VARCHAR(50),  -- battery, network, storage, cpu, memory, custom
    unit VARCHAR(20),
    data_type VARCHAR(20),  -- int, float, string, bool
    source VARCHAR(50),  -- xsight, mobicontrol, custom
    is_standard BIT DEFAULT 0,
    validation_rules NVARCHAR(MAX)  -- JSON: min, max, allowed_values
);
CREATE INDEX idx_source ON metric_definitions(source);
CREATE INDEX idx_category ON metric_definitions(category);
CREATE INDEX idx_standard ON metric_definitions(is_standard);
PRINT 'Table metric_definitions created.';
GO

-- Telemetry Points (time-series data)
-- NOTE: For production, implement partitioning by timestamp
CREATE TABLE telemetry_points (
    id BIGINT IDENTITY(1,1),
    device_id VARCHAR(50) NOT NULL,
    tenant_id VARCHAR(50) NOT NULL,
    timestamp DATETIME2 NOT NULL,
    metric_id VARCHAR(50) NOT NULL,
    value FLOAT,
    value_str NVARCHAR(255),
    quality INT DEFAULT 100,
    ingestion_time DATETIME2 DEFAULT GETUTCDATE(),
    source_batch_id VARCHAR(50),
    CONSTRAINT pk_telemetry PRIMARY KEY (id, timestamp),
    FOREIGN KEY (device_id) REFERENCES devices(device_id),
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id),
    FOREIGN KEY (metric_id) REFERENCES metric_definitions(metric_id)
);
CREATE INDEX idx_device_time ON telemetry_points(device_id, timestamp);
CREATE INDEX idx_tenant_time ON telemetry_points(tenant_id, timestamp);
CREATE INDEX idx_metric_time ON telemetry_points(metric_id, timestamp);
CREATE INDEX idx_ingestion ON telemetry_points(ingestion_time);
PRINT 'Table telemetry_points created.';
GO

-- Baselines (normal behavior profiles)
CREATE TABLE baselines (
    baseline_id VARCHAR(50) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    name NVARCHAR(255) NOT NULL,
    scope VARCHAR(50) NOT NULL,  -- tenant, site, device_group, device
    scope_id VARCHAR(50) NOT NULL,
    metric_id VARCHAR(50) NOT NULL,
    window_config NVARCHAR(MAX),  -- JSON: window_days, update_frequency, time_segmentation
    stats NVARCHAR(MAX),  -- JSON: mean, std, median, p5, p95
    valid_from DATETIME2 NOT NULL,
    valid_to DATETIME2,
    created_by VARCHAR(50),
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id),
    FOREIGN KEY (metric_id) REFERENCES metric_definitions(metric_id)
);
CREATE INDEX idx_tenant_scope ON baselines(tenant_id, scope, scope_id);
CREATE INDEX idx_valid ON baselines(valid_from, valid_to);
CREATE INDEX idx_metric ON baselines(metric_id);
PRINT 'Table baselines created.';
GO

-- Anomalies (detected anomalies with metadata)
CREATE TABLE anomalies (
    anomaly_id VARCHAR(50) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    device_id VARCHAR(50) NOT NULL,
    event_id VARCHAR(50),  -- Link to anomaly_events table
    timestamp DATETIME2 NOT NULL,
    detector_name VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,  -- low, medium, high, critical
    score FLOAT NOT NULL,
    metrics_involved NVARCHAR(MAX),  -- JSON: {metric_id: value}
    explanation NVARCHAR(MAX),
    explanation_cache_key VARCHAR(100),
    user_feedback VARCHAR(20),  -- true_positive, false_positive, unknown
    status VARCHAR(20) DEFAULT 'new',  -- new, acknowledged, resolved, ignored
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    updated_at DATETIME2 DEFAULT GETUTCDATE(),
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id),
    FOREIGN KEY (device_id) REFERENCES devices(device_id)
);
CREATE INDEX idx_tenant_time_anomaly ON anomalies(tenant_id, timestamp DESC);
CREATE INDEX idx_device_time_anomaly ON anomalies(device_id, timestamp DESC);
CREATE INDEX idx_severity ON anomalies(severity, timestamp DESC);
CREATE INDEX idx_status ON anomalies(status, timestamp DESC);
CREATE INDEX idx_detector ON anomalies(detector_name);
CREATE INDEX idx_feedback ON anomalies(user_feedback);
CREATE INDEX idx_event_id ON anomalies(event_id);
PRINT 'Table anomalies created.';
GO

-- Change Log (environmental changes)
CREATE TABLE change_log (
    change_id VARCHAR(50) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    timestamp DATETIME2 NOT NULL,
    change_type VARCHAR(50) NOT NULL,  -- policy, os_version, app_version, ap_added
    description NVARCHAR(MAX),
    affected_devices NVARCHAR(MAX),  -- JSON: list of device_ids
    source VARCHAR(50),
    metadata NVARCHAR(MAX),  -- JSON
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id)
);
CREATE INDEX idx_tenant_time_change ON change_log(tenant_id, timestamp DESC);
CREATE INDEX idx_type_time ON change_log(change_type, timestamp DESC);
CREATE INDEX idx_source_change ON change_log(source);
PRINT 'Table change_log created.';
GO

-- Explanations (LLM-generated explanations)
CREATE TABLE explanations (
    explanation_id VARCHAR(50) PRIMARY KEY,
    anomaly_id VARCHAR(50) NOT NULL,
    llm_model VARCHAR(100) NOT NULL,
    prompt_version VARCHAR(20),
    generated_text NVARCHAR(MAX) NOT NULL,
    confidence FLOAT,
    tokens_used INT,
    generation_time_ms INT,
    context_used NVARCHAR(MAX),  -- JSON: Retrieved RAG context
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    FOREIGN KEY (anomaly_id) REFERENCES anomalies(anomaly_id)
);
CREATE INDEX idx_anomaly_explain ON explanations(anomaly_id);
CREATE INDEX idx_model ON explanations(llm_model);
CREATE INDEX idx_created_explain ON explanations(created_at);
PRINT 'Table explanations created.';
GO

-- Audit Logs (compliance and security)
CREATE TABLE audit_logs (
    log_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    timestamp DATETIME2 DEFAULT GETUTCDATE() NOT NULL,
    user_id VARCHAR(50),
    tenant_id VARCHAR(50) NOT NULL,
    action VARCHAR(50) NOT NULL,  -- view, create, update, delete, export
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(50),
    ip_address VARCHAR(45),
    metadata NVARCHAR(MAX)  -- JSON: request details
);
CREATE INDEX idx_timestamp_audit ON audit_logs(timestamp);
CREATE INDEX idx_user_audit ON audit_logs(user_id, timestamp);
CREATE INDEX idx_tenant_audit ON audit_logs(tenant_id, timestamp);
CREATE INDEX idx_resource ON audit_logs(resource_type, resource_id);
PRINT 'Table audit_logs created.';
GO

-- ============================================================================
-- New Tables: Events, Patterns, ML Models, Alerting
-- ============================================================================

-- Anomaly Events (Group consecutive anomalies into events)
CREATE TABLE anomaly_events (
    event_id VARCHAR(50) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    device_id VARCHAR(50) NOT NULL,
    event_start DATETIME2 NOT NULL,
    event_end DATETIME2 NOT NULL,
    duration_minutes INT NOT NULL,
    anomaly_score_min FLOAT NOT NULL,
    anomaly_score_max FLOAT NOT NULL,
    anomaly_score_mean FLOAT NOT NULL,
    row_count INT NOT NULL,
    severity VARCHAR(20) NOT NULL,  -- low, medium, high, critical
    metrics_json NVARCHAR(MAX),  -- JSON: aggregated metric values
    status VARCHAR(20) DEFAULT 'new',  -- new, acknowledged, resolved, ignored
    model_version VARCHAR(50),
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id),
    FOREIGN KEY (device_id) REFERENCES devices(device_id)
);
CREATE INDEX idx_ev_tenant_time ON anomaly_events(tenant_id, event_start DESC);
CREATE INDEX idx_ev_device_time ON anomaly_events(device_id, event_start DESC);
CREATE INDEX idx_ev_severity ON anomaly_events(severity, event_start DESC);
CREATE INDEX idx_ev_status ON anomaly_events(status, event_start DESC);
PRINT 'Table anomaly_events created.';
GO

-- Device Patterns (Long-term device behavior patterns)
CREATE TABLE device_patterns (
    pattern_id VARCHAR(50) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    device_id VARCHAR(50) NOT NULL,
    period_start DATETIME2 NOT NULL,
    period_end DATETIME2 NOT NULL,
    total_points INT NOT NULL,
    total_anomalies INT NOT NULL,
    anomaly_rate FLOAT NOT NULL,
    event_count INT NOT NULL,
    worst_anomaly_score FLOAT NOT NULL,
    mean_anomaly_score FLOAT NOT NULL,
    pattern_json NVARCHAR(MAX),  -- JSON: detailed pattern data
    explanation NVARCHAR(MAX),  -- LLM-generated pattern explanation
    model_version VARCHAR(50),
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id),
    FOREIGN KEY (device_id) REFERENCES devices(device_id)
);
CREATE INDEX idx_pat_tenant_device ON device_patterns(tenant_id, device_id, period_start DESC);
CREATE INDEX idx_pat_period ON device_patterns(period_start, period_end);
CREATE INDEX idx_pat_anomaly_rate ON device_patterns(anomaly_rate DESC);
PRINT 'Table device_patterns created.';
GO

-- ML Models (Registry of trained anomaly detection models)
CREATE TABLE ml_models (
    model_id VARCHAR(50) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    name NVARCHAR(255) NOT NULL,
    model_type VARCHAR(50) NOT NULL,  -- isolation_forest, z_score, hybrid, etc.
    version VARCHAR(50) NOT NULL,
    config_json NVARCHAR(MAX),  -- JSON: model configuration
    feature_cols_json NVARCHAR(MAX),  -- JSON: list of feature columns used
    model_artifact VARBINARY(MAX),  -- Serialized model (optional, can store externally)
    status VARCHAR(20) DEFAULT 'training',  -- training, trained, deployed, archived
    trained_at DATETIME2,
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id)
);
CREATE INDEX idx_model_tenant ON ml_models(tenant_id);
CREATE INDEX idx_model_type ON ml_models(model_type);
CREATE INDEX idx_model_status ON ml_models(status);
CREATE INDEX idx_model_version ON ml_models(version);
PRINT 'Table ml_models created.';
GO

-- Model Deployments (Track which models are deployed where)
CREATE TABLE model_deployments (
    deployment_id VARCHAR(50) PRIMARY KEY,
    model_id VARCHAR(50) NOT NULL,
    environment VARCHAR(50) NOT NULL,  -- production, staging, development
    is_active BIT DEFAULT 1,
    deployed_at DATETIME2 DEFAULT GETUTCDATE(),
    deployed_by VARCHAR(50),  -- user_id
    FOREIGN KEY (model_id) REFERENCES ml_models(model_id)
);
CREATE INDEX idx_deploy_model ON model_deployments(model_id);
CREATE INDEX idx_deploy_env ON model_deployments(environment, is_active);
CREATE INDEX idx_deploy_active ON model_deployments(is_active, deployed_at DESC);
PRINT 'Table model_deployments created.';
GO

-- Model Metrics (Performance metrics for deployed models)
CREATE TABLE model_metrics (
    metric_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    model_id VARCHAR(50) NOT NULL,
    timestamp DATETIME2 DEFAULT GETUTCDATE() NOT NULL,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    true_positives INT DEFAULT 0,
    false_positives INT DEFAULT 0,
    false_negatives INT DEFAULT 0,
    total_predictions INT DEFAULT 0,
    extra_data NVARCHAR(MAX),  -- JSON: additional metrics
    FOREIGN KEY (model_id) REFERENCES ml_models(model_id)
);
CREATE INDEX idx_metrics_model_time ON model_metrics(model_id, timestamp DESC);
CREATE INDEX idx_metrics_timestamp ON model_metrics(timestamp);
PRINT 'Table model_metrics created.';
GO

-- Alert Rules (Configurable alerting conditions)
CREATE TABLE alert_rules (
    rule_id VARCHAR(50) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    name NVARCHAR(255) NOT NULL,
    rule_type VARCHAR(50) NOT NULL,  -- anomaly_severity, anomaly_count, pattern_detected, etc.
    conditions_json NVARCHAR(MAX) NOT NULL,  -- JSON: rule conditions
    severity VARCHAR(20) NOT NULL,  -- low, medium, high, critical
    actions_json NVARCHAR(MAX),  -- JSON: notification actions (email, webhook, etc.)
    is_enabled BIT DEFAULT 1,
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    updated_at DATETIME2 DEFAULT GETUTCDATE(),
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id)
);
CREATE INDEX idx_rule_tenant ON alert_rules(tenant_id);
CREATE INDEX idx_rule_enabled ON alert_rules(is_enabled, tenant_id);
CREATE INDEX idx_rule_type ON alert_rules(rule_type);
PRINT 'Table alert_rules created.';
GO

-- Alerts (Generated alerts from rules)
CREATE TABLE alerts (
    alert_id VARCHAR(50) PRIMARY KEY,
    rule_id VARCHAR(50) NOT NULL,
    tenant_id VARCHAR(50) NOT NULL,
    anomaly_id VARCHAR(50),  -- Optional: link to specific anomaly
    device_id VARCHAR(50),  -- Optional: link to device
    event_id VARCHAR(50),  -- Optional: link to anomaly event
    severity VARCHAR(20) NOT NULL,  -- low, medium, high, critical
    status VARCHAR(20) DEFAULT 'open',  -- open, acknowledged, resolved, suppressed
    message NVARCHAR(MAX) NOT NULL,
    triggered_at DATETIME2 DEFAULT GETUTCDATE(),
    acknowledged_at DATETIME2,
    acknowledged_by VARCHAR(50),  -- user_id
    resolved_at DATETIME2,
    extra_data NVARCHAR(MAX),  -- JSON: additional alert context
    FOREIGN KEY (rule_id) REFERENCES alert_rules(rule_id),
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id),
    FOREIGN KEY (anomaly_id) REFERENCES anomalies(anomaly_id),
    FOREIGN KEY (device_id) REFERENCES devices(device_id)
);
CREATE INDEX idx_alert_tenant_time ON alerts(tenant_id, triggered_at DESC);
CREATE INDEX idx_alert_status ON alerts(status, triggered_at DESC);
CREATE INDEX idx_alert_severity ON alerts(severity, triggered_at DESC);
CREATE INDEX idx_alert_rule ON alerts(rule_id);
CREATE INDEX idx_alert_device ON alerts(device_id, triggered_at DESC);
PRINT 'Table alerts created.';
GO

-- Alert Notifications (Notification delivery tracking)
CREATE TABLE alert_notifications (
    notification_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    alert_id VARCHAR(50) NOT NULL,
    channel VARCHAR(50) NOT NULL,  -- email, webhook, sms, slack, etc.
    recipient NVARCHAR(255) NOT NULL,  -- email address, webhook URL, phone number, etc.
    status VARCHAR(20) DEFAULT 'pending',  -- pending, sent, failed, delivered
    response_json NVARCHAR(MAX),  -- JSON: response from notification service
    sent_at DATETIME2,
    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id)
);
CREATE INDEX idx_notif_alert ON alert_notifications(alert_id);
CREATE INDEX idx_notif_status ON alert_notifications(status, sent_at);
CREATE INDEX idx_notif_channel ON alert_notifications(channel);
PRINT 'Table alert_notifications created.';
GO

-- ============================================================================
-- Cost Intelligence Module Tables
-- ============================================================================

-- Device Type Costs (Hardware asset costs by device model)
CREATE TABLE device_type_costs (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    device_model NVARCHAR(255) NOT NULL,  -- Device model string
    currency_code VARCHAR(3) DEFAULT 'USD' NOT NULL,
    purchase_cost BIGINT NOT NULL,  -- Cost in cents
    replacement_cost BIGINT,  -- Cost in cents
    repair_cost_avg BIGINT,  -- Cost in cents
    depreciation_months INT DEFAULT 36,
    residual_value_percent INT DEFAULT 0,
    warranty_months INT,
    valid_from DATETIME2 DEFAULT GETUTCDATE() NOT NULL,
    valid_to DATETIME2,  -- NULL = currently active
    notes NVARCHAR(MAX),
    created_by NVARCHAR(100),
    created_at DATETIME2 DEFAULT GETUTCDATE() NOT NULL,
    updated_at DATETIME2 DEFAULT GETUTCDATE() NOT NULL,
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id)
);
CREATE INDEX idx_dtc_tenant_model ON device_type_costs(tenant_id, device_model);
CREATE INDEX idx_dtc_tenant_valid ON device_type_costs(tenant_id, valid_from, valid_to);
CREATE INDEX idx_dtc_active ON device_type_costs(tenant_id, valid_to);
CREATE INDEX idx_dtc_currency ON device_type_costs(tenant_id, currency_code);
PRINT 'Table device_type_costs created.';
GO

-- Operational Costs (Custom operational cost entries per tenant)
CREATE TABLE operational_costs (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    name NVARCHAR(255) NOT NULL,
    description NVARCHAR(MAX),
    category VARCHAR(50) NOT NULL,  -- labor, downtime, support, infrastructure, maintenance, other
    currency_code VARCHAR(3) DEFAULT 'USD' NOT NULL,
    amount BIGINT NOT NULL,  -- Cost in cents
    cost_type VARCHAR(50) NOT NULL,  -- hourly, daily, per_incident, fixed_monthly, per_device
    unit NVARCHAR(50),  -- hour, day, incident, month, device
    scope_type VARCHAR(50) DEFAULT 'tenant',  -- tenant, location, device_group, device_model
    scope_id NVARCHAR(100),  -- ID of the scope entity (null for tenant-wide)
    is_active BIT DEFAULT 1 NOT NULL,
    valid_from DATETIME2 DEFAULT GETUTCDATE() NOT NULL,
    valid_to DATETIME2,  -- NULL = currently active
    notes NVARCHAR(MAX),
    created_by NVARCHAR(100),
    created_at DATETIME2 DEFAULT GETUTCDATE() NOT NULL,
    updated_at DATETIME2 DEFAULT GETUTCDATE() NOT NULL,
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id)
);
CREATE INDEX idx_oc_tenant_name ON operational_costs(tenant_id, name);
CREATE INDEX idx_oc_tenant_category ON operational_costs(tenant_id, category);
CREATE INDEX idx_oc_tenant_valid ON operational_costs(tenant_id, valid_from, valid_to);
CREATE INDEX idx_oc_active ON operational_costs(tenant_id, is_active);
CREATE INDEX idx_oc_scope ON operational_costs(tenant_id, scope_type, scope_id);
PRINT 'Table operational_costs created.';
GO

-- Cost Audit Logs (Change tracking for compliance)
CREATE TABLE cost_audit_logs (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,  -- device_type_cost, operational_cost
    entity_id BIGINT NOT NULL,
    action VARCHAR(20) NOT NULL,  -- create, update, delete
    old_values_json NVARCHAR(MAX),  -- JSON: Previous values
    new_values_json NVARCHAR(MAX),  -- JSON: New values
    changed_fields_json NVARCHAR(MAX),  -- JSON array of changed field names
    user_id NVARCHAR(100),
    user_email NVARCHAR(255),
    change_reason NVARCHAR(MAX),
    source VARCHAR(50) DEFAULT 'manual',  -- manual, api, import, system
    ip_address VARCHAR(45),
    user_agent NVARCHAR(500),
    request_id NVARCHAR(100),
    timestamp DATETIME2 DEFAULT GETUTCDATE() NOT NULL,
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id)
);
CREATE INDEX idx_cal_tenant_time ON cost_audit_logs(tenant_id, timestamp);
CREATE INDEX idx_cal_entity ON cost_audit_logs(entity_type, entity_id, timestamp);
CREATE INDEX idx_cal_user ON cost_audit_logs(user_id, timestamp);
CREATE INDEX idx_cal_action ON cost_audit_logs(tenant_id, action, timestamp);
PRINT 'Table cost_audit_logs created.';
GO

-- Cost Calculation Cache (Pre-computed cost impacts)
CREATE TABLE cost_calculation_cache (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,  -- anomaly, anomaly_event, insight, device
    entity_id NVARCHAR(100) NOT NULL,
    device_id VARCHAR(50),
    device_model NVARCHAR(255),
    currency_code VARCHAR(3) DEFAULT 'USD' NOT NULL,
    hardware_cost BIGINT DEFAULT 0,  -- Cost in cents
    downtime_cost BIGINT DEFAULT 0,
    labor_cost BIGINT DEFAULT 0,
    other_cost BIGINT DEFAULT 0,
    total_cost BIGINT DEFAULT 0,
    potential_savings BIGINT DEFAULT 0,
    investment_required BIGINT DEFAULT 0,
    payback_months FLOAT,
    breakdown_json NVARCHAR(MAX),  -- JSON: Detailed breakdown
    impact_level VARCHAR(20),  -- high, medium, low
    confidence_score FLOAT DEFAULT 0.7,
    confidence_explanation NVARCHAR(MAX),
    calculation_version VARCHAR(20) DEFAULT 'v1',
    cost_config_snapshot_json NVARCHAR(MAX),
    calculated_at DATETIME2 DEFAULT GETUTCDATE() NOT NULL,
    expires_at DATETIME2,
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id)
);
CREATE INDEX idx_ccc_tenant_entity ON cost_calculation_cache(tenant_id, entity_type, entity_id);
CREATE INDEX idx_ccc_device ON cost_calculation_cache(tenant_id, device_id);
CREATE INDEX idx_ccc_calculated ON cost_calculation_cache(tenant_id, calculated_at);
CREATE INDEX idx_ccc_impact ON cost_calculation_cache(tenant_id, impact_level);
PRINT 'Table cost_calculation_cache created.';
GO

-- Add foreign key constraints (Must be done after all tables are created)
-- Foreign key for anomaly_events in anomalies table
ALTER TABLE anomalies
ADD CONSTRAINT fk_anomaly_event FOREIGN KEY (event_id) REFERENCES anomaly_events(event_id);
GO

-- Foreign key for event_id in alerts table
ALTER TABLE alerts
ADD CONSTRAINT fk_alert_event FOREIGN KEY (event_id) REFERENCES anomaly_events(event_id);
GO

-- ============================================================================
-- Seed Data (Development/Testing)
-- ============================================================================

PRINT 'Inserting seed data...';
GO

-- Insert default tenant
IF NOT EXISTS (SELECT * FROM tenants WHERE tenant_id = 'default')
BEGIN
    INSERT INTO tenants (tenant_id, name, tier) VALUES ('default', 'Default Tenant', 'standard');
    PRINT 'Default tenant created.';
END
GO

-- Insert standard metric definitions
IF NOT EXISTS (SELECT * FROM metric_definitions WHERE metric_id = 'battery_drop')
BEGIN
    INSERT INTO metric_definitions (metric_id, name, category, unit, data_type, source, is_standard)
    VALUES
        ('battery_drop', 'Battery Level Drop', 'battery', '%', 'int', 'xsight', 1),
        ('free_storage', 'Free Storage', 'storage', 'KB', 'int', 'xsight', 1),
        ('download', 'Download', 'network', 'bytes', 'int', 'xsight', 1),
        ('upload', 'Upload', 'network', 'bytes', 'int', 'xsight', 1),
        ('offline_time', 'Offline Time', 'connectivity', 'minutes', 'int', 'xsight', 1),
        ('disconnect_count', 'Disconnect Count', 'connectivity', 'count', 'int', 'xsight', 1),
        ('wifi_signal', 'WiFi Signal Strength', 'network', 'dBm', 'int', 'xsight', 1),
        ('connection_time', 'Connection Time', 'network', 'minutes', 'int', 'xsight', 1);
    PRINT 'Standard metric definitions created.';
END
GO

-- Insert additional metric definitions for events and patterns
IF NOT EXISTS (SELECT * FROM metric_definitions WHERE metric_id = 'anomaly_event_count')
BEGIN
    INSERT INTO metric_definitions (metric_id, name, category, unit, data_type, source, is_standard)
    VALUES
        ('anomaly_event_count', 'Anomaly Event Count', 'anomaly', 'count', 'int', 'system', 1),
        ('anomaly_rate', 'Anomaly Rate', 'anomaly', '%', 'float', 'system', 1),
        ('pattern_score', 'Pattern Score', 'pattern', 'score', 'float', 'system', 1);
    PRINT 'Additional metric definitions created.';
END
GO

-- Insert default alert rules (optional seed data)
IF NOT EXISTS (SELECT * FROM alert_rules WHERE rule_id = 'critical_severity_default')
BEGIN
    INSERT INTO alert_rules (rule_id, tenant_id, name, rule_type, conditions_json, severity, actions_json, is_enabled)
    VALUES
        ('critical_severity_default', 'default', 'Critical Severity Anomaly', 'anomaly_severity',
         '{"severity": "critical"}', 'critical',
         '{"channels": ["email"], "recipients": []}', 1),
        ('high_severity_default', 'default', 'High Severity Anomaly', 'anomaly_severity',
         '{"severity": "high", "count_threshold": 5}', 'high',
         '{"channels": ["email"], "recipients": []}', 1);
    PRINT 'Default alert rules created.';
END
GO

PRINT '';
PRINT '============================================================================';
PRINT 'Backend database schema created successfully!';
PRINT '';
PRINT 'Database: SOTI_AnomalyDetection';
PRINT 'Tables created: 22';
PRINT '  Core: tenants, users, devices';
PRINT '  Data: metric_definitions, telemetry_points';
PRINT '  Analysis: baselines, anomalies, anomaly_events, device_patterns';
PRINT '  ML: ml_models, model_deployments, model_metrics';
PRINT '  Alerting: alert_rules, alerts, alert_notifications';
PRINT '  Cost Intelligence: device_type_costs, operational_costs, cost_audit_logs, cost_calculation_cache';
PRINT '  System: change_log, explanations, audit_logs';
PRINT '';
PRINT 'Default tenant created: default';
PRINT 'Standard metrics: 11';
PRINT 'Default alert rules: 2';
PRINT '';
PRINT 'Next steps:';
PRINT '  1. Create admin user';
PRINT '  2. Register devices from XSight/MobiControl';
PRINT '  3. Start ingesting telemetry data';
PRINT '  4. Train and deploy ML models';
PRINT '  5. Configure alert rules and recipients';
PRINT '  6. Configure cost data in Cost Intelligence module';
PRINT '============================================================================';
GO
