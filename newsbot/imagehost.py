"""Where post images go so LINE can show them.

LINE only accepts images by public HTTPS URL. On GitHub Actions the images are
committed to a separate `cards` branch of this (public) repo and served from
raw.githubusercontent.com. Anywhere else they are just saved to out/cards/.
"""
from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

import requests

from .config import ROOT

log = logging.getLogger("newsbot")
API = "https://api.github.com"


class ImageHostError(RuntimeError):
    pass


class LocalImageHost:
    """Saves images to disk. Returns no URL, so LINE gets no picture."""

    def __init__(self, folder: Path | None = None):
        self.folder = folder or ROOT / "out"   # paths already start with cards/
        self.note = (f"บันทึกรูปไว้ที่ {self.folder / 'cards'} "
                     "(ไม่ได้ส่งเข้า LINE เพราะไม่ได้รันบน GitHub)")

    def upload(self, path: str, data: bytes) -> str | None:
        target = self.folder / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return None


class GitHubImageHost:
    def __init__(self, repo: str, token: str, branch: str = "cards", timeout: float = 20,
                 session: requests.Session | None = None):
        self.repo = repo
        self.branch = branch
        self.timeout = timeout
        self.http = session or requests.Session()
        self.http.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        self._ready = False
        self.note = ""

    def _call(self, method: str, url: str, **kw) -> requests.Response:
        resp = self.http.request(method, API + url, timeout=self.timeout, **kw)
        if resp.status_code >= 400 and resp.status_code != 404:
            raise ImageHostError(f"GitHub {method} {url}: HTTP {resp.status_code} {resp.text[:200]}")
        return resp

    def _prepare(self) -> None:
        if self._ready:
            return
        info = self._call("GET", f"/repos/{self.repo}")
        if info.status_code == 404:
            raise ImageHostError(f"ไม่พบ repo {self.repo}")
        meta = info.json()
        if meta.get("private"):
            raise ImageHostError("repo ยังเป็น Private ทำให้ LINE เปิดรูปไม่ได้ "
                                 "เปลี่ยนเป็น Public ที่ Settings → Danger Zone → Change visibility")
        ref = self._call("GET", f"/repos/{self.repo}/git/ref/heads/{self.branch}")
        if ref.status_code == 404:
            base = meta.get("default_branch", "main")
            sha = self._call("GET", f"/repos/{self.repo}/git/ref/heads/{base}").json()["object"]["sha"]
            self._call("POST", f"/repos/{self.repo}/git/refs",
                       json={"ref": f"refs/heads/{self.branch}", "sha": sha})
        self._ready = True

    def upload(self, path: str, data: bytes) -> str:
        self._prepare()
        resp = self._call("PUT", f"/repos/{self.repo}/contents/{path}", json={
            "message": f"card: {path}",
            "content": base64.b64encode(data).decode("ascii"),
            "branch": self.branch,
        })
        if resp.status_code == 404:
            raise ImageHostError(f"อัปโหลดรูปไม่ได้ ({path})")
        return f"https://raw.githubusercontent.com/{self.repo}/{self.branch}/{path}"


def make_image_host(branch: str = "cards"):
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if repo and token:
        return GitHubImageHost(repo, token, branch)
    return LocalImageHost()
