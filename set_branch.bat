@echo off
chcp 65001 >nul
echo ============================================================
echo GitHub 仓库分支管理
echo ============================================================
echo.

set /p TOKEN="请输入您的 GitHub Personal Access Token (需要 repo 权限): "

if "%TOKEN%"=="" (
    echo 错误: 未提供 GitHub token
    exit /b 1
)

echo.
echo 正在将默认分支设置为: MeaCap_InvLM_origin...

powershell -Command "$headers = @{'Authorization'='token %TOKEN%'; 'Accept'='application/vnd.github.v3+json'}; $body = @{default_branch='MeaCap_InvLM_origin'} | ConvertTo-Json; try { Invoke-RestMethod -Uri 'https://api.github.com/repos/chenaying/meacap_project' -Method PATCH -Headers $headers -Body $body -ContentType 'application/json' | Out-Null; Write-Host '成功将默认分支设置为: MeaCap_InvLM_origin' -ForegroundColor Green } catch { Write-Host '设置默认分支失败:' -ForegroundColor Red; Write-Host $_.Exception.Message -ForegroundColor Red; exit 1 }"

if errorlevel 1 (
    echo 操作失败
    exit /b 1
)

echo.
echo 正在删除远程分支: MeaCap_origin...
git push origin --delete MeaCap_origin 2>&1

if errorlevel 1 (
    echo 警告: 删除分支可能失败（可能已不存在）
) else (
    echo 成功删除远程分支: MeaCap_origin
)

echo.
echo ============================================================
echo 操作完成！
echo   - 默认分支已设置为: MeaCap_InvLM_origin
echo   - 已尝试删除旧分支: MeaCap_origin
echo ============================================================

