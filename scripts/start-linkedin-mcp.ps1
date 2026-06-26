# Start the LinkedIn MCP HTTP sidecar for HermesHire job search.
# Prerequisites: uv installed — https://docs.astral.sh/uv/getting-started/installation/
# One-time login: uvx mcp-server-linkedin@latest --login

$ErrorActionPreference = "Stop"

$hostAddr = if ($env:LINKEDIN_MCP_HOST) { $env:LINKEDIN_MCP_HOST } else { "127.0.0.1" }
$port = if ($env:LINKEDIN_MCP_PORT) { $env:LINKEDIN_MCP_PORT } else { "8080" }
$path = if ($env:LINKEDIN_MCP_PATH) { $env:LINKEDIN_MCP_PATH } else { "/mcp" }

Write-Host "Starting LinkedIn MCP sidecar at http://${hostAddr}:${port}${path}"
Write-Host "Press Ctrl+C to stop."

uvx mcp-server-linkedin@latest `
  --transport streamable-http `
  --host $hostAddr `
  --port $port `
  --path $path
