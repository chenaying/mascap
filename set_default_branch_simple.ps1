# Simple script to set default branch using GitHub API

$REPO_OWNER = "chenaying"
$REPO_NAME = "meacap_project"
$NEW_DEFAULT_BRANCH = "MeaCap_InvLM_origin"
$OLD_BRANCH = "MeaCap_origin"

Write-Host "============================================================"
Write-Host "GitHub Repository Branch Management"
Write-Host "============================================================"
Write-Host ""

# Get GitHub token
$token = Read-Host -Prompt "Enter your GitHub Personal Access Token (requires repo permission)"

if ([string]::IsNullOrWhiteSpace($token)) {
    Write-Host "Error: GitHub token not provided" -ForegroundColor Red
    exit 1
}

# GitHub API URL
$apiUrl = "https://api.github.com/repos/$REPO_OWNER/$REPO_NAME"

# Set request headers
$headers = @{
    "Authorization" = "token $token"
    "Accept" = "application/vnd.github.v3+json"
}

# Step 1: Set default branch
Write-Host "Setting default branch to: $NEW_DEFAULT_BRANCH..." -ForegroundColor Yellow
try {
    $body = @{
        default_branch = $NEW_DEFAULT_BRANCH
    } | ConvertTo-Json
    
    $response = Invoke-RestMethod -Uri $apiUrl -Method PATCH -Headers $headers -Body $body -ContentType "application/json"
    
    Write-Host "Successfully set default branch to: $NEW_DEFAULT_BRANCH" -ForegroundColor Green
} catch {
    Write-Host "Failed to set default branch" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 2: Delete old branch
Write-Host "Deleting remote branch: $OLD_BRANCH..." -ForegroundColor Yellow
$deleteResult = git push origin --delete $OLD_BRANCH 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "Successfully deleted remote branch: $OLD_BRANCH" -ForegroundColor Green
} else {
    Write-Host "Warning: Could not delete branch (may already be deleted)" -ForegroundColor Yellow
    Write-Host "Output: $deleteResult"
}

Write-Host ""
Write-Host "Operation completed!" -ForegroundColor Green
Write-Host "  - Default branch set to: $NEW_DEFAULT_BRANCH"
Write-Host "  - Attempted to delete old branch: $OLD_BRANCH"

