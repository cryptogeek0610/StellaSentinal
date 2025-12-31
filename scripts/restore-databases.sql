-- SQL Server Database Restore Script
-- Restores customer databases from BAK files for multi-tenant training
--
-- Databases:
--   BENELUX: XSight_BENELUX, MobiControl_BENELUX
--   PIBLIC:  XSight_PIBLIC, MobiControl_PIBLIC
--
-- Usage: Run this script after SQL Server container is healthy
--        sqlcmd -S localhost -U sa -P 'YourPassword' -i restore-databases.sql

USE master;
GO

-- ============================================================================
-- BENELUX Customer Databases
-- ============================================================================

-- XSight Database (BENELUX)
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'XSight_BENELUX')
BEGIN
    PRINT 'Restoring XSight_BENELUX...';

    RESTORE DATABASE [XSight_BENELUX]
    FROM DISK = '/var/opt/mssql/backup/BENELUX/20251113_110933_SOTI_XSight_dw.bak'
    WITH
        MOVE 'SOTI_XSight_dw' TO '/var/opt/mssql/data/XSight_BENELUX.mdf',
        MOVE 'SOTI_XSight_dw_log' TO '/var/opt/mssql/data/XSight_BENELUX_log.ldf',
        REPLACE,
        STATS = 10;

    PRINT 'XSight_BENELUX restored successfully.';
END
ELSE
BEGIN
    PRINT 'XSight_BENELUX already exists, skipping restore.';
END
GO

-- MobiControl Database (BENELUX)
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'MobiControl_BENELUX')
BEGIN
    PRINT 'Restoring MobiControl_BENELUX...';

    RESTORE DATABASE [MobiControl_BENELUX]
    FROM DISK = '/var/opt/mssql/backup/BENELUX/MC-BackupYW.bak'
    WITH
        MOVE 'MobiControlDB' TO '/var/opt/mssql/data/MobiControl_BENELUX.mdf',
        MOVE 'MobiControlDB_log' TO '/var/opt/mssql/data/MobiControl_BENELUX_log.ldf',
        REPLACE,
        STATS = 10;

    PRINT 'MobiControl_BENELUX restored successfully.';
END
ELSE
BEGIN
    PRINT 'MobiControl_BENELUX already exists, skipping restore.';
END
GO

-- ============================================================================
-- PIBLIC Customer Databases
-- ============================================================================

-- XSight Database (PIBLIC)
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'XSight_PIBLIC')
BEGIN
    PRINT 'Restoring XSight_PIBLIC...';

    RESTORE DATABASE [XSight_PIBLIC]
    FROM DISK = '/var/opt/mssql/backup/PIBLIC/S106800-SOTI_XSight_dw-20250917194338.bak'
    WITH
        MOVE 'SOTI_XSight_dw' TO '/var/opt/mssql/data/XSight_PIBLIC.mdf',
        MOVE 'SOTI_XSight_dw_log' TO '/var/opt/mssql/data/XSight_PIBLIC_log.ldf',
        REPLACE,
        STATS = 10;

    PRINT 'XSight_PIBLIC restored successfully.';
END
ELSE
BEGIN
    PRINT 'XSight_PIBLIC already exists, skipping restore.';
END
GO

-- MobiControl Database (PIBLIC)
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'MobiControl_PIBLIC')
BEGIN
    PRINT 'Restoring MobiControl_PIBLIC...';

    RESTORE DATABASE [MobiControl_PIBLIC]
    FROM DISK = '/var/opt/mssql/backup/PIBLIC/S106800-MobiControlDB-20250917193048.bak'
    WITH
        MOVE 'MobiControlDB' TO '/var/opt/mssql/data/MobiControl_PIBLIC.mdf',
        MOVE 'MobiControlDB_log' TO '/var/opt/mssql/data/MobiControl_PIBLIC_log.ldf',
        REPLACE,
        STATS = 10;

    PRINT 'MobiControl_PIBLIC restored successfully.';
END
ELSE
BEGIN
    PRINT 'MobiControl_PIBLIC already exists, skipping restore.';
END
GO

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
WHERE name IN ('XSight_BENELUX', 'MobiControl_BENELUX', 'XSight_PIBLIC', 'MobiControl_PIBLIC')
ORDER BY name;

PRINT '';
PRINT 'All customer databases are ready for training!';
GO
