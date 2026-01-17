# GitHub 仓库分支管理脚本
# 使用 GitHub API 设置默认分支并删除旧分支

$REPO_OWNER = "chenaying"
$REPO_NAME = "meacap_project"
$NEW_DEFAULT_BRANCH = "MeaCap_InvLM_origin"
$OLD_BRANCH = "MeaCap_origin"

Write-Host ("=" * 60)
Write-Host "GitHub 仓库分支管理"
Write-Host ("=" * 60)
Write-Host ""

# 获取 GitHub token
$token = Read-Host -Prompt "请输入您的 GitHub Personal Access Token (需要 repo 权限)" -AsSecureString
$tokenPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($token))

if ([string]::IsNullOrWhiteSpace($tokenPlain)) {
    Write-Host "错误: 未提供 GitHub token" -ForegroundColor Red
    exit 1
}

# GitHub API 基础 URL
$apiBaseUrl = "https://api.github.com/repos/$REPO_OWNER/$REPO_NAME"

# 设置请求头
$headers = @{
    "Authorization" = "token $tokenPlain"
    "Accept" = "application/vnd.github.v3+json"
}

# 1. 设置默认分支
Write-Host "正在将默认分支设置为: $NEW_DEFAULT_BRANCH..." -ForegroundColor Yellow
try {
    $body = @{
        default_branch = $NEW_DEFAULT_BRANCH
    } | ConvertTo-Json
    
    $response = Invoke-RestMethod -Uri $apiBaseUrl -Method PATCH -Headers $headers -Body $body -ContentType "application/json"
    
    Write-Host "✓ 成功将默认分支设置为: $NEW_DEFAULT_BRANCH" -ForegroundColor Green
} catch {
    Write-Host "✗ 设置默认分支失败" -ForegroundColor Red
    Write-Host "  错误信息: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""

# 2. 删除旧分支
Write-Host "正在删除远程分支: $OLD_BRANCH..." -ForegroundColor Yellow
try {
    $deleteResult = git push origin --delete $OLD_BRANCH 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ 成功删除远程分支: $OLD_BRANCH" -ForegroundColor Green
    } else {
        Write-Host "✗ 删除远程分支失败" -ForegroundColor Red
        Write-Host "  输出: $deleteResult" -ForegroundColor Red
        Write-Host ""
        Write-Host "警告: 可能该分支已不存在，或需要手动删除" -ForegroundColor Yellow
    }
} catch {
    Write-Host "✗ 删除分支时出错: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "✓ 操作完成!" -ForegroundColor Green
Write-Host "  - 默认分支已设置为: $NEW_DEFAULT_BRANCH" -ForegroundColor Cyan
Write-Host "  - 已尝试删除旧分支: $OLD_BRANCH" -ForegroundColor Cyan

