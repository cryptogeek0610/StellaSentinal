============================================================
  SOTI Stella Sentinel - Deployment Package
============================================================
  Last Updated: 2025-12-29
============================================================

This folder contains everything needed to deploy the Stella
Sentinel anomaly detection application on a new machine.

PREREQUISITES
------------------------------------------------------------
1. Docker Desktop installed and running
   Download: https://www.docker.com/products/docker-desktop/

2. Minimum 8GB RAM available for Docker
   (Recommended: 16GB for optimal performance)

3. Network access to your SOTI XSight SQL Server database


QUICK START (Windows)
------------------------------------------------------------
1. Copy this entire folder to the target machine
2. Double-click install.bat
3. Follow the on-screen prompts to configure your environment
4. Access the application at http://localhost:3000


MANUAL INSTALLATION
------------------------------------------------------------
1. Copy env.template to .env:
   copy env.template .env

2. Edit .env and configure:
   - BACKEND_DB_PASS: Password for PostgreSQL (change from default)
   - DW_DB_HOST: Your XSight SQL Server hostname
   - DW_DB_USER: SQL Server username
   - DW_DB_PASS: SQL Server password
   - Optional: MC_DB_* for MobiControl database
   - Optional: LLM_* for AI-powered explanations

3. Build Docker images:
   docker-compose build

4. Start the application:
   docker-compose up -d

5. Access the application:
   - Frontend: http://localhost:3000
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs


SERVICES INCLUDED
------------------------------------------------------------
- app:        Python API backend (anomaly detection engine)
- scheduler:  ML training scheduler (autonomous model training)
- ml-worker:  ML worker (handles training jobs)
- frontend:   React web interface
- postgres:   PostgreSQL database (application data)
- redis:      Redis (job queue)
- ollama:     Ollama LLM (optional, for AI explanations)
- qdrant:     Qdrant vector database (RAG support)


USEFUL COMMANDS
------------------------------------------------------------
View logs:            docker-compose logs -f
View specific logs:   docker-compose logs -f app
Stop application:     docker-compose down
Restart application:  docker-compose restart
Restart one service:  docker-compose restart app
View status:          docker-compose ps

Pull Ollama model (for AI explanations):
  docker exec stellasentinel-ollama ollama pull llama3.2


FOLDER STRUCTURE
------------------------------------------------------------
ApplicationReady/
  install.bat         - Windows installation script
  env.template        - Environment configuration template
  docker-compose.yml  - Docker orchestration config
  Dockerfile          - Python backend image definition
  pyproject.toml      - Python dependencies
  src/                - Python application source code
  frontend/           - React frontend source code
    Dockerfile        - Frontend image definition
    nginx.conf        - Nginx web server configuration


TROUBLESHOOTING
------------------------------------------------------------
1. "Port already in use" error:
   Edit .env and change the port (e.g., API_PORT=8001)

2. Cannot connect to SQL Server:
   - Verify DW_DB_HOST is correct
   - Check firewall allows port 1433
   - Try DW_TRUST_SERVER_CERT=true for self-signed certs

3. Out of memory:
   - Increase Docker memory limit in Docker Desktop settings
   - Stop unused containers: docker-compose down

4. Slow startup:
   - First build takes longer (downloading images)
   - Wait 30-60 seconds for all services to start


SUPPORT
------------------------------------------------------------
For issues and feature requests, contact your system administrator
or refer to the project documentation.
