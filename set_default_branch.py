#!/usr/bin/env python3
"""
使用 GitHub API 设置默认分支并删除旧分支
"""

import requests
import subprocess
import sys
import getpass

# GitHub 仓库信息
REPO_OWNER = "chenaying"
REPO_NAME = "meacap_project"
NEW_DEFAULT_BRANCH = "MeaCap_InvLM_origin"
OLD_BRANCH = "MeaCap_origin"

def get_github_token():
    """获取 GitHub token"""
    token = getpass.getpass("请输入您的 GitHub Personal Access Token (需要 repo 权限): ")
    return token.strip()

def set_default_branch(token, branch):
    """设置默认分支"""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "default_branch": branch
    }
    
    print(f"正在将默认分支设置为: {branch}...")
    response = requests.patch(url, headers=headers, json=data)
    
    if response.status_code == 200:
        print(f"✓ 成功将默认分支设置为: {branch}")
        return True
    else:
        print(f"✗ 设置默认分支失败: {response.status_code}")
        print(f"  错误信息: {response.text}")
        return False

def delete_remote_branch(branch):
    """删除远程分支"""
    print(f"正在删除远程分支: {branch}...")
    result = subprocess.run(
        ["git", "push", "origin", "--delete", branch],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✓ 成功删除远程分支: {branch}")
        return True
    else:
        print(f"✗ 删除远程分支失败:")
        print(f"  {result.stderr}")
        return False

def main():
    print("=" * 60)
    print("GitHub 仓库分支管理")
    print("=" * 60)
    print()
    
    # 获取 token
    token = get_github_token()
    if not token:
        print("错误: 未提供 GitHub token")
        sys.exit(1)
    
    # 设置默认分支
    if not set_default_branch(token, NEW_DEFAULT_BRANCH):
        print("\n错误: 无法设置默认分支")
        sys.exit(1)
    
    print()
    
    # 删除旧分支
    if not delete_remote_branch(OLD_BRANCH):
        print(f"\n警告: 无法删除分支 {OLD_BRANCH}")
        print("   可能该分支已不存在，或需要手动删除")
    else:
        print(f"\n✓ 所有操作完成!")
        print(f"  - 默认分支已设置为: {NEW_DEFAULT_BRANCH}")
        print(f"  - 已删除旧分支: {OLD_BRANCH}")

if __name__ == "__main__":
    main()

