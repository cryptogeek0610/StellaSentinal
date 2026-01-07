-- SQL Server Database Restore Script
-- Restores XSight_DW database from BAK file
--
-- NOTE: MobiControlDB backup requires SQL Server 2025 which is not available on ARM.
--       Only XSight_DW (SQL Server 2022 backup) is supported.
--
-- Usage:
--   1. Place XSight_DW.bak in ./backups/ directory
--   2. Run: docker exec stellasentinal-sqlserver /opt/mssql-tools18/bin/sqlcmd \
--           -S localhost -U sa -P 'YourPassword' -C -i /var/opt/mssql/scripts/restore-databases.sql
--
-- Or use the setup script: ./scripts/setup-sqlserver.sh

USE master;
GO

-- ============================================================================
-- XSight_DW Database (Telemetry Data)
-- ============================================================================

IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'XSight_DW')
BEGIN
    PRINT 'Restoring XSight_DW...';

    RESTORE DATABASE [XSight_DW]
    FROM DISK = '/var/opt/mssql/backup/XSight_DW.bak'
    WITH
        MOVE 'SOTI_XSight_dw' TO '/var/opt/mssql/data/XSight_DW.mdf',
        MOVE 'SOTI_XSight_dw_2023_02' TO '/var/opt/mssql/data/XSight_DW_2023_02.mdf',
        MOVE 'SOTI_XSight_dw_2023_03' TO '/var/opt/mssql/data/XSight_DW_2023_03.mdf',
        MOVE 'SOTI_XSight_dw_2023_04' TO '/var/opt/mssql/data/XSight_DW_2023_04.mdf',
        MOVE 'SOTI_XSight_dw_log' TO '/var/opt/mssql/data/XSight_DW_log.ldf',
        REPLACE,
        STATS = 10;

    PRINT 'XSight_DW restored successfully.';
END
ELSE
BEGIN
    PRINT 'XSight_DW already exists, skipping restore.';
END
GO

-- ============================================================================
-- MobiControlDB - NOT SUPPORTED
-- ============================================================================
-- MobiControlDB backup requires SQL Server 2025 (version 998)
-- SQL Server 2022 (version 957) cannot restore this backup.
-- To use MobiControlDB, you would need to:
-- 1. Run SQL Server 2025 on an x86 machine (not ARM/Apple Silicon)
-- 2. Or request a backup from SQL Server 2022 or earlier

-- ============================================================================
-- Verification
-- ============================================================================

PRINT '';
PRINT '=== Database Restore Summary ===';

SELECT
    name AS DatabaseName,
    state_desc AS State,
    create_date AS CreatedDate
FROM sys.databases
WHERE name = 'XSight_DW'
ORDER BY name;

PRINT '';
PRINT 'XSight_DW database is ready!';
GO
